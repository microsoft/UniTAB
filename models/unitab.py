from typing import Dict, Optional
import random
import json
import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

import util.dist as dist
from util import box_ops
from util.metrics import accuracy
from util.misc import NestedTensor, interpolate

from .backbone import build_backbone
from .postprocessors import build_postprocessors
from .transformer_unitab import build_transformer


def withbbox_subseq(textonly_subseq, bbox, start_token, end_token):
    withbbox_subseq = torch.zeros(textonly_subseq.shape[0]+4+2).long().to(textonly_subseq.device)
    withbbox_subseq[0] = start_token
    withbbox_subseq[1:len(textonly_subseq)+1] = textonly_subseq
    withbbox_subseq[-5:-1] = bbox
    withbbox_subseq[-1] = end_token
    return withbbox_subseq

def target2prevind(caption_idx, targets, num_bins=None, text_vocab=50265):
    obj_start, obj_end = text_vocab, text_vocab+1
    bs, max_length = caption_idx.shape
    return_idx = torch.ones(caption_idx.shape).long().to(caption_idx.device)
    ## not batch for now due to varied number of bbox
    for bi in range(bs):
        target_bbox = targets[bi]['boxes'].clone() ## otherwise do not converge
        if target_bbox.shape[1]!=4:
            return_idx[bi,:]=caption_idx[bi,:]  ## no bbox
            continue
        target_bbox[:,:2], target_bbox[:,2:] = target_bbox[:,:2] - target_bbox[:,2:]/2, target_bbox[:,:2] + target_bbox[:,2:]/2 ## (x1,y1,x2,y2)
        target_bbox = ((target_bbox*(num_bins-1)).long()+2+text_vocab).clip(text_vocab+2,text_vocab+num_bins+1)
        positive_map = targets[bi]['positive_map']
        positive_map_idx = (positive_map!=0).nonzero(as_tuple=True)
        begin_tokens, end_tokens, convert_seq = [], [], []
        for obj_i in range(target_bbox.shape[0]):
            token_idx = positive_map_idx[1][positive_map_idx[0]==obj_i]
            if token_idx.shape[0]==0:   ## two samples in refcocog training has box with no corresponding words. Skip as follow, or append box in the end.
                continue
            begin_idx, end_idx = int(torch.min(token_idx)), int(torch.max(token_idx))+1
            span = int(1./positive_map[obj_i,token_idx[0]])
            if span!=(end_idx-begin_idx):
                ## from end, later span
                all_begin_idx, all_end_idx = begin_idx, end_idx
                begin_idx, end_idx = all_end_idx-1, all_end_idx
                while positive_map[obj_i,begin_idx]!=0:
                    begin_idx-=1
                begin_idx += 1
                ## end of temporaty solution
                begin_tokens.append(begin_idx)
                end_tokens.append(end_idx)
                convert_seq.append(withbbox_subseq(caption_idx[bi,begin_idx:end_idx],target_bbox[obj_i,:],obj_start,obj_end))
                ## from begin, first span
                begin_idx, end_idx = all_begin_idx, all_begin_idx
                while positive_map[obj_i,end_idx]!=0:
                    end_idx+=1
                ## end of temporaty solution
                begin_tokens.append(begin_idx)
                end_tokens.append(end_idx)
                convert_seq.append(withbbox_subseq(caption_idx[bi,begin_idx:end_idx],target_bbox[obj_i,:],obj_start,obj_end))
            else:
                begin_tokens.append(begin_idx)
                end_tokens.append(end_idx)
                convert_seq.append(withbbox_subseq(caption_idx[bi,begin_idx:end_idx],target_bbox[obj_i,:],obj_start,obj_end))

        tmp_return_idx = caption_idx[bi,:].clone()
        for idx, (begin, end, clipidx) in enumerate(sorted(zip(begin_tokens, end_tokens, list(range(len(begin_tokens)))),reverse=True)):
            tmp_return_idx = [tmp_return_idx[:begin],convert_seq[clipidx],tmp_return_idx[end:]]
            tmp_return_idx = torch.cat(tmp_return_idx,dim=0)
        return_idx[bi,:]=tmp_return_idx[:max_length]
    return return_idx

def target2gtind(caption_idx, targets, num_bins=None, text_vocab=50265):
    return_idx = target2prevind(caption_idx, targets,num_bins=num_bins,text_vocab=text_vocab)
    return_idx = torch.cat([return_idx[:,1:],torch.ones(return_idx.shape[0],1).long().to(return_idx.device)],dim=1)
    return return_idx


def outputsclass_2_predbboxes(bbox_logits, pred_bbox=None, num_bins=None, do_flickrgrounding=None, do_refexp=None, text_vocab=50265):
    if do_refexp: ## tmp solution for merged RefExp.
        return_bbox = torch.zeros(bbox_logits.shape[0],4).to(bbox_logits.device)
        pred_bbox = bbox_logits.argmax(dim=-1)  ## (x1,y1,x2,y2)
        for bi in range(bbox_logits.shape[0]):
            end_indx = torch.where(pred_bbox[bi,:]==(text_vocab+1))[0]
            if end_indx.shape[0]==0: continue
            end_indx = int(end_indx[0])
            if end_indx<4: continue
            bbox_i = pred_bbox[bi,end_indx-4:end_indx]
            bbox_i = (bbox_i-2-text_vocab).float()/(num_bins-1)
            bbox_i[:2], bbox_i[2:] = (bbox_i[:2]+bbox_i[2:])/2, (bbox_i[2:]-bbox_i[:2])  ##(x_c, y_c, w,h)
            return_bbox[bi,:] = bbox_i
        return return_bbox.clip(min=0).unsqueeze(1)
    elif not do_flickrgrounding:
        ## return placeholder of fixed number of box
        ## placeholder to match format of bbox prediction; not used
        pred_bbox = bbox_logits.argmax(dim=-1)[:,:4]  ## (x1,y1,x2,y2)
        pred_bbox = (pred_bbox-2).float()/(num_bins-1)
        pred_bbox[:,:2], pred_bbox[:,2:] = (pred_bbox[:,:2]+pred_bbox[:,2:])/2, (pred_bbox[:,2:]-pred_bbox[:,:2])  ##(x_c, y_c, w,h)
        return pred_bbox.clip(min=0).unsqueeze(1)
    else:
        ## return all concerted boxes; index processed later in models/postprocessors.py
        return_bbox = torch.zeros(bbox_logits.shape[0],100,4).float().to(bbox_logits.device)
        if pred_bbox is None:
            pred_bbox = bbox_logits.argmax(dim=-1)  ## (x1,y1,x2,y2)
        for bi in range(pred_bbox.shape[0]):
            bbox_list = []
            for wi in range(pred_bbox.shape[1]):
                if pred_bbox[bi,wi]==text_vocab+1 and wi>=4:
                    bbox_list.append((pred_bbox[bi,wi-4:wi]-2-text_vocab).float()/(num_bins-1))
            if len(bbox_list)!=0:
                return_bbox_ii = torch.stack(bbox_list,dim=0)[:100,:]
                return_bbox[bi,:return_bbox_ii.shape[0],:] = return_bbox_ii
        return_bbox[:,:,:2], return_bbox[:,:,2:] = \
            (return_bbox[:,:,:2]+return_bbox[:,:,2:])/2, (return_bbox[:,:,2:]-return_bbox[:,:,:2])  ##(x_c, y_c, w,h)
        return return_bbox.clip(min=0)

class UniTAB(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        max_decoding_step=None,
        do_flickrgrounding=None,
        do_refexp=None
    ):
        super().__init__()
        self.max_decoding_step = max_decoding_step
        self.do_flickrgrounding = do_flickrgrounding
        self.do_refexp = do_refexp
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.vocab_size = self.transformer.embedding.word_embeddings.weight.shape[0]
        self.text_vocab=len(self.transformer.tokenizer)
        assert(self.vocab_size==self.text_vocab or self.vocab_size==self.text_vocab+num_queries+2)
        self.coordclass_embed = nn.Linear(hidden_dim, self.vocab_size)   ## reuse for coord regression
        self.coordclass_embed.weight = self.transformer.embedding.word_embeddings.weight
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor, captions, targets, encode_and_save=True, memory_cache=None):
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)
        if encode_and_save:
            assert memory_cache is None
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()
            coordquery_embed = self.transformer.embedding.word_embeddings.weight
            memory_cache = self.transformer(
                self.input_proj(src),
                mask,
                coordquery_embed,
                pos[-1],
                captions,
                encode_and_save=True,
                text_memory=None,
                img_memory=None,
                text_attention_mask=None,
            )

            return memory_cache

        else:
            assert memory_cache is not None
            pred_seq = None

            if self.training:
                prev_indx = torch.stack([targets[ii]['previdx_gt'] for ii in range(len(targets))]).squeeze(1).to(memory_cache["tokenized"]['input_ids'].device)
                hs = self.transformer(
                    mask=memory_cache["mask"],
                    prev_indx=prev_indx,
                    pos_embed=memory_cache["pos_embed"],
                    encode_and_save=False,
                    text_memory=memory_cache["text_memory_resized"],
                    img_memory=memory_cache["img_memory"],
                    text_attention_mask=memory_cache["text_attention_mask"],
                )
                outputs_class = self.coordclass_embed(hs[-1,:,:,:])
                pred_seq = outputs_class.argmax(dim=-1)
            else:   ## inference
                # greedy decoding at test time
                bs = memory_cache["tokenized"]['input_ids'].shape[0]
                prev_indx = torch.zeros(bs,self.max_decoding_step).long().to(memory_cache["tokenized"]['input_ids'].device)
                pred_seq = torch.zeros(bs,self.max_decoding_step).long().to(memory_cache["tokenized"]['input_ids'].device)
                dec_step_num = prev_indx.shape[1]

                early_stop = torch.zeros(prev_indx.shape[0]).to(memory_cache["tokenized"]['input_ids'].device)
                for t in range(dec_step_num):
                    hs = self.transformer(
                        mask=memory_cache["mask"],
                        prev_indx=prev_indx,
                        pos_embed=memory_cache["pos_embed"],
                        encode_and_save=False,
                        text_memory=memory_cache["text_memory_resized"],
                        img_memory=memory_cache["img_memory"],
                        text_attention_mask=memory_cache["text_attention_mask"],
                    )
                    outputs_class = self.coordclass_embed(hs[-1,:,:,:])
                    argmax_inds = outputs_class.argmax(dim=-1)
                    if (early_stop!=0).all(): break   ## early stop
                    early_stop = early_stop + (argmax_inds[:,t]==2).long()
                    prev_indx[:, 1:] = argmax_inds[:, :-1]
                    pred_seq = argmax_inds
            pred_boxes = outputsclass_2_predbboxes(outputs_class, pred_bbox=pred_seq, \
                num_bins=self.num_queries,do_flickrgrounding=self.do_flickrgrounding,do_refexp=self.do_refexp) ## any random value, not used anyway
            out = {}
            out.update(
                {
                    "pred_logits": outputs_class,
                    "pred_seq": pred_seq,
                    "pred_boxes": pred_boxes,
                }
            )
            out["caption_gt"] = torch.stack([targets[ii]['target_gt'] for ii in range(len(targets))]).squeeze(1).to(memory_cache["tokenized"]['input_ids'].device)
            out["image_id"] = [x["image_id"] for x in targets]
            out["sentence_id"] = [x["sentence_id"] for x in targets]
            out["original_img_id"] = [x["original_img_id"] for x in targets]
            return out

class SetCaptionCriterion(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, outputs, targets, positive_map):
        target_indx = outputs["caption_gt"]
        logits = outputs["pred_logits"].permute(0,2,1)#.log_softmax(-1)  # BS x (num_queries) x (num_tokens)
        torch.set_deterministic(False)
        loss_ce = F.cross_entropy(
            logits, target_indx
            , ignore_index=1
        )
        torch.set_deterministic(True)
        losses = {"loss_ce": loss_ce}
        return losses

def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = UniTAB(
        backbone,
        transformer,
        num_queries=args.num_queries,
        max_decoding_step=args.max_decoding_step,
        do_flickrgrounding=(args.do_flickrgrounding), ## used only in seq2box decoding
        do_refexp=('refcoco' in args.dataset_config or 'multitask' in args.dataset_config), ## refcocog is the default eval for multitask finetune
    )
    weight_dict = {"loss_ce": 1.}

    criterion = None
    criterion = SetCaptionCriterion()
    criterion.to(device)

    return model, criterion, weight_dict

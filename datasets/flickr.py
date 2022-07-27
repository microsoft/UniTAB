from pathlib import Path

from transformers import RobertaTokenizerFast

from .coco import ModulatedDetection, make_coco_transforms

import torch
import random
import numpy as np
import sys
import sys
sys.path.append("..")
from models.unitab import target2prevind as target2prevind_caption
from models.unitab import target2gtind as target2gtind_caption

class FlickrDetection(ModulatedDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, return_tokens, tokenizer, is_train=False,\
        max_decoding_step=256, num_queries=200, do_flickrgrounding=None, unitab_pretrain=False, pretrain_seqcrop=None,\
        multitask=False, GT_type='', refexp_dataset_name=''):
        super(FlickrDetection, self).__init__(img_folder, ann_file, transforms, return_masks, return_tokens, tokenizer, is_train)
        self.vqav2 = 'vqav2caption' in str(ann_file)
        self.tokenizer = tokenizer
        self.max_decoding_step = max_decoding_step
        self.num_queries = num_queries
        self.do_flickrgrounding = do_flickrgrounding
        self.unitab_pretrain = unitab_pretrain
        self.pretrain_seqcrop = pretrain_seqcrop
        self.multitask = multitask
        self.GT_type = GT_type
        self.refexp_dataset_name = refexp_dataset_name

    def __getitem__(self, idx):
        img, target = super(FlickrDetection, self).__getitem__(idx)
        caption, output_caption = '', target['caption']

        # ## overwrite box annptations
        pad_idx = 1 ## RoBert
        if self.multitask:  ## do grounding or captioning
            if random.random()<0.5: target['caption'] = output_caption
            else: target['caption'] = caption
        elif self.vqav2:
            qas = target['caption'].split('<VQAQA>')
            assert(len(qas)==11)
            caption, output_caption = qas[0],random.choice(qas[1:])
            target['caption'] = caption
        elif self.do_flickrgrounding:
            target['caption'] = output_caption
        else:
            target['caption'] = caption

        tokenized = self.tokenizer.batch_encode_plus([output_caption], padding="max_length", \
            max_length=self.max_decoding_step, truncation=True, return_tensors="pt")
        target_gt = target2gtind_caption(tokenized['input_ids'], \
            [target],num_bins=self.num_queries)
        previdx_gt = target2prevind_caption(tokenized['input_ids'], \
            [target],num_bins=self.num_queries)
        target['target_gt'] = target_gt
        target['target_gt_textonly'] = torch.cat([tokenized['input_ids'][:,1:],pad_idx*torch.ones(tokenized['input_ids'].shape[0],1).long()],dim=1)
        target['previdx_gt'] = previdx_gt
        target['previdx_gt_textonly'] = tokenized['input_ids']
        if self.unitab_pretrain and self.is_train:
            ind_list = []
            in_obj=0
            for ii in range(previdx_gt.shape[-1]):
                if previdx_gt[0,ii]==1: break
                if previdx_gt[0,ii]==50265: in_obj+=1
                if previdx_gt[0,ii]==50266: in_obj-=1
                if previdx_gt[0,ii]<50265 and in_obj==0: ind_list.append(ii)
            if self.pretrain_seqcrop=='first': split_idx = ind_list[0]
            elif self.pretrain_seqcrop=='random' or self.pretrain_seqcrop=='rand':
                split_idx = random.choice(ind_list)
            elif self.pretrain_seqcrop=='grounding':
                target['caption'] = output_caption
                return img, target
            elif self.pretrain_seqcrop=='mixed':
                if random.random()<0.5: split_idx = ind_list[0]
                else:
                    target['caption'] = tokenized
                    return img, target
            else:
                print('split_idx undefined in flickr.py')
                exit(0)

            tokenized_input = self.tokenizer.batch_encode_plus([''], padding="max_length", \
                max_length=self.max_decoding_step, truncation=True, return_tensors="pt")    ## placeholder
            split_input_tokenized = previdx_gt[0,:split_idx+1]
            if split_input_tokenized[-1]!=2:
                split_input_tokenized = torch.cat([split_input_tokenized,2*torch.ones(1).long()],dim=0)
            split_input_tokenized = split_input_tokenized[split_input_tokenized<50265].unsqueeze(0)

            input_leng = split_input_tokenized.shape[-1]
            tokenized_input['input_ids'] = torch.cat([split_input_tokenized,torch.ones(1,self.max_decoding_step-input_leng).long()],dim=1)
            tokenized_input['attention_mask'] = torch.cat([torch.ones(1,input_leng).long(),torch.zeros(1,self.max_decoding_step-input_leng).long()],dim=1)
            target['caption'] = tokenized_input

            ## update decoder GTs
            if split_idx!=0:
                target_gt = torch.cat([target_gt[:,split_idx:], torch.ones(1,self.max_decoding_step-target_gt[0,split_idx:].shape[0]).long()],dim=1)
                previdx_gt = torch.cat([torch.zeros(1,1).long(), target_gt[:,:-1]],dim=1)
            target['target_gt'] = target_gt
            target['previdx_gt'] = previdx_gt
        return img, target


def build(image_set, args):

    img_dir = Path(args.flickr_img_path) / f"{image_set}"
    if args.test and args.GT_type == "merged": ## flickr grounding test
        img_dir = Path(args.flickr_img_path) / "test"

    if args.GT_type == "merged":
        identifier = "mergedGT"
    elif args.GT_type == "separate":
        identifier = "separateGT"
    elif args.GT_type == "mergedGT_pretrain":
        identifier = "mergedGT_pretrain"
    elif args.GT_type == "merged_karpathy":
        identifier = "mergedGT_karpathy"
        img_dir = Path(args.flickr_img_path)
    elif args.GT_type == "mscococaption":
        identifier = "mscococaption"
        img_dir = Path(args.flickr_img_path) / f"all2014"
    elif args.GT_type == "vqav2caption":
        identifier = "vqav2caption" ## VQA-std
        img_dir = Path(args.flickr_img_path) / f"{image_set}2014"
    elif args.GT_type == "vqav2captionKP":
        identifier = "vqav2captionKP" ## VQA-KP
        img_dir = Path(args.flickr_img_path) / f"all2014"
    else:
        assert False, f"{args.GT_type} is not a valid type of annotation for flickr"

    ## pretraining file that exclude both original and KP splits' val & test
    if args.unitab_pretrain and (image_set=="train") and args.GT_type == "merged":
        identifier = "mergedGT_pretrain"
    elif args.unitab_pretrain and (image_set=="train") and args.GT_type == "separate":
        identifier = "separateGT_pretrain"

    if args.test:
        ann_file = Path(args.flickr_ann_path) / f"final_flickr_{identifier}_test.json"
    else:
        ann_file = Path(args.flickr_ann_path) / f"final_flickr_{identifier}_{image_set}.json"

    ## trainval & test-dev split
    if args.GT_type == "vqav2caption":
        if image_set=='train':
            ann_file = Path(args.flickr_ann_path) / f"final_flickr_{identifier}_trainval.json"
            img_dir = Path(args.flickr_img_path) / f"all2014"
        else:
            ann_file = Path(args.flickr_ann_path) / f"final_flickr_{identifier}_test2015.json"
            img_dir = Path(args.flickr_img_path) / f"test2015"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = FlickrDetection(
        img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=False,
        return_tokens=True,  # args.contrastive_align_loss,
        tokenizer=tokenizer,
        is_train=image_set=="train",
        max_decoding_step=args.max_decoding_step,
        num_queries=args.num_queries,
        do_flickrgrounding=args.do_flickrgrounding,
        unitab_pretrain=args.unitab_pretrain,
        pretrain_seqcrop=args.pretrain_seqcrop,
        multitask=(not args.unitab_pretrain and args.GT_type == "mergedGT_pretrain"), ## MTL finetune
        GT_type=args.GT_type,
        refexp_dataset_name=args.refexp_dataset_name,
    )
    return dataset

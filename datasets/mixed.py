import os
import os.path
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets.vision import VisionDataset
from transformers import RobertaTokenizerFast

from .coco import ConvertCocoPolysToMask, make_coco_transforms

import torch
import random
import numpy as np
import sys
sys.path.append("..")
from models.unitab import target2prevind as target2prevind_caption
from models.unitab import target2gtind as target2gtind_caption

class CustomCocoDetection(VisionDataset):
    def __init__(
        self,
        root_coco: str,
        root_vg: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super(CustomCocoDetection, self).__init__(root_coco, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root_coco = root_coco
        self.root_vg = root_vg

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        dataset = img_info["data_source"] if "data_source" in img_info else "coco"

        cur_root = self.root_coco if dataset == "coco" else self.root_vg
        img = Image.open(os.path.join(cur_root, path)).convert("RGB")
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)

class MixedDetection(CustomCocoDetection):
    """Same as the modulated detection dataset, except with multiple img sources"""

    def __init__(self, img_folder_coco, img_folder_vg, ann_file, transforms, return_masks, return_tokens, tokenizer, is_train=False,\
        max_decoding_step=256, num_queries=200, unitab_pretrain=False, pretrain_seqcrop=None):
        super(MixedDetection, self).__init__(img_folder_coco, img_folder_vg, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, return_tokens, tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.max_decoding_step = max_decoding_step
        self.num_queries = num_queries
        self.unitab_pretrain = unitab_pretrain
        self.pretrain_seqcrop = pretrain_seqcrop
        self.is_train = is_train

    def __getitem__(self, idx):
        img, target = super(MixedDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        caption = self.coco.loadImgs(image_id)[0]["caption"]
        target = {"image_id": image_id, "annotations": target, "caption": caption}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        ## added for unitab_pretrain
        if self.unitab_pretrain and self.is_train:
            ## output_text with bbox version
            tokenized = self.tokenizer.batch_encode_plus([target['caption']], padding="max_length", \
                max_length=self.max_decoding_step, truncation=True, return_tensors="pt")
            target_gt = target2gtind_caption(tokenized['input_ids'], \
                [target],num_bins=self.num_queries)
            previdx_gt = target2prevind_caption(tokenized['input_ids'], \
                [target],num_bins=self.num_queries)
            ## to match flickr format (in eval though)
            target['sentence_id'] = torch.ones(1).long()*-1
            target['original_img_id'] = torch.ones(1).long()*-1

            target['target_gt'] = target_gt
            target['target_gt_textonly'] = torch.cat([tokenized['input_ids'][:,1:],torch.ones(tokenized['input_ids'].shape[0],1).long()],dim=1)
            target['previdx_gt'] = previdx_gt
            target['previdx_gt_textonly'] = tokenized['input_ids']

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
                print('split_idx undefined in mixed.py')
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
    vg_img_dir = Path(args.vg_img_path)
    coco_img_dir = Path(args.coco_path) / f"{image_set}2014"
    assert vg_img_dir.exists(), f"provided VG img path {vg_img_dir} does not exist"
    assert coco_img_dir.exists(), f"provided coco img path {coco_img_dir} does not exist"

    ann_file = Path(args.gqa_ann_path) / f"final_mixed_{image_set}.json"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = MixedDetection(
        coco_img_dir,
        vg_img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=args.masks,
        return_tokens=True,
        tokenizer=tokenizer,
        is_train=image_set=="train",
        max_decoding_step=args.max_decoding_step,
        num_queries=args.num_queries,
        unitab_pretrain=args.unitab_pretrain,
        pretrain_seqcrop=args.pretrain_seqcrop
    )

    return dataset

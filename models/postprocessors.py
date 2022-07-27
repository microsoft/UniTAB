# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Postprocessors class to transform MDETR output according to the downstream task"""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops

def seq2logits(bbox_logits, text_vocab=50265):
    return_bbox = torch.zeros(bbox_logits.shape[0],100,4).float().to(bbox_logits.device)
    return_logits = torch.zeros(bbox_logits.shape[0],100,256).float().to(bbox_logits.device)
    pred_bbox = bbox_logits.argmax(dim=-1)  ## (x1,y1,x2,y2)
    for bi in range(pred_bbox.shape[0]):
        str_list, bbox_list, box_idx = [], [], []
        in_bbox = 0
        for wi in range(pred_bbox.shape[1]):
            if pred_bbox[bi,wi]==text_vocab:
                in_bbox+=1
            elif pred_bbox[bi,wi]==text_vocab+1:
                in_bbox-=1
            elif pred_bbox[bi,wi]>text_vocab+1:
                bbox_list.append(pred_bbox[bi,wi])
            else:
                str_list.append(pred_bbox[bi,wi])
                if in_bbox: box_idx.append(len(bbox_list)//4)
                else: box_idx.append(-1)
        while len(bbox_list)%4!=0:
            bbox_list.append(text_vocab) ## any special token
        bbox_list = torch.tensor(bbox_list).view(-1,4).float()
        for wi in range(len(box_idx)):
            if box_idx[wi]!=-1 and box_idx[wi]<bbox_list.shape[0]:
                return_logits[bi, box_idx[wi], wi+1]=1. ## start token
    return return_logits

class PostProcessFlickr(nn.Module):
    """This module converts the model's output for Flickr30k entities evaluation.

    This processor is intended for recall@k evaluation with respect to each phrase in the sentence.
    It requires a description of each phrase (as a binary mask), and returns a sorted list of boxes for each phrase.
    """

    @torch.no_grad()
    def forward(self, outputs, target_sizes, positive_map, items_per_batch_element):
        """Perform the computation.
        Args:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            positive_map: tensor [total_nbr_phrases x max_seq_len] for each phrase in the batch, contains a binary
                          mask of the tokens that correspond to that sentence. Note that this is a "collapsed" batch,
                          meaning that all the phrases of all the batch elements are stored sequentially.
            items_per_batch_element: list[int] number of phrases corresponding to each batch element.
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        out_logits = seq2logits(out_logits)

        batch_size = target_sizes.shape[0]

        prob = F.softmax(out_logits, -1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # and from relative [0, 1] to absolute [0, height] coordinates
        boxes = boxes * scale_fct[:, None, :]

        cum_sum = np.cumsum(items_per_batch_element)

        curr_batch_index = 0
        # binarize the map if not already binary
        pos = positive_map > 1e-6

        predicted_boxes = [[] for _ in range(batch_size)]

        # The collapsed batch dimension must match the number of items
        assert len(pos) == cum_sum[-1]

        if len(pos) == 0:
            return predicted_boxes

        # if the first batch elements don't contain elements, skip them.
        while cum_sum[curr_batch_index] == 0:
            curr_batch_index += 1

        for i in range(len(pos)):
            # scores are computed by taking the max over the scores assigned to the positive tokens
            scores, _ = torch.max(pos[i].unsqueeze(0) * prob[curr_batch_index, :, :], dim=-1)
            _, indices = torch.sort(scores, descending=True)

            assert items_per_batch_element[curr_batch_index] > 0
            predicted_boxes[curr_batch_index].append(boxes[curr_batch_index][indices].to("cpu").tolist())
            if i == len(pos) - 1:
                break

            # check if we need to move to the next batch element
            while i >= cum_sum[curr_batch_index] - 1:
                curr_batch_index += 1
                assert curr_batch_index < len(cum_sum)

        return predicted_boxes

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        out_logits = seq2logits(out_logits)

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        labels = torch.ones_like(labels)

        scores = 1 - prob[:, :, -1]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        assert len(scores) == len(labels) == len(boxes)
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results

def build_postprocessors(args, dataset_name) -> Dict[str, nn.Module]:
    postprocessors: Dict[str, nn.Module] = {"bbox": PostProcess()}
    if dataset_name == "flickr":
        postprocessors["flickr_bbox"] = PostProcessFlickr()

    return postprocessors

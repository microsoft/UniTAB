# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .mixed import CustomCocoDetection
from .coco import build as build_coco
from .flickr import build as build_flickr
from .mixed import build as build_mixed
from .refexp import build as build_refexp

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, (torchvision.datasets.CocoDetection, CustomCocoDetection)):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == "coco":
        return build_coco(image_set, args)
    if dataset_file == "flickr":
        return build_flickr(image_set, args)
    if dataset_file == "mixed":
        return build_mixed(image_set, args)
    if dataset_file == "refexp":
        return build_refexp(image_set, args)
    raise ValueError(f"dataset {dataset_file} not supported")

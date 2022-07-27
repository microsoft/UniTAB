# UniTAB: Unifying Text and Box Outputs for Grounded VL Modeling
[UniTAB: Unifying Text and Box Outputs for Grounded Vision-Language Modeling](https://arxiv.org/pdf/2111.12085.pdf)

by [Zhengyuan Yang](https://zhengyuan.info), [Zhe Gan](https://zhegan27.github.io/), [Jianfeng Wang](http://jianfengwang.me/), [Xiaowei Hu](https://scholar.google.com/citations?user=Pj0TwxwAAAAJ&hl=en), [Faisal Ahmed](https://scholar.google.com/citations?hl=en&user=laKl8acAAAAJ), [Zicheng Liu](https://zicliu.wixsite.com/mysite), [Yumao Lu](https://www.linkedin.com/in/yumao/), [Lijuan Wang](https://www.microsoft.com/en-us/research/people/lijuanw/)

European Conference on Computer Vision, 2022, Oral Presentation


### Introduction
We propose UniTAB, a vision-language (VL) model that unifies text generation and bounding box prediction into a single architecture.
For more details, please refer to our
[paper](https://arxiv.org/pdf/2111.12085.pdf).


<p align="center">
  <img src="https://zyang-ur.github.io//unitab/unitab.jpg" width="100%"/>
</p>

### Citation

    @inproceedings{yang2022unitab,
      title={UniTAB: Unifying Text and Box Outputs for Grounded Vision-Language Modeling},
      author={Yang, Zhengyuan and Gan, Zhe and Wang, Jianfeng and Hu, Xiaowei and Ahmed, Faisal and Liu, Zicheng and Lu, Yumao and Wang, Lijuan},
      booktitle={ECCV},
      year={2022}
    }


## Installation

Clone the repository:
```
git clone https://github.com/microsoft/UniTAB.git
cd UniTAB
```

New conda env:
```
conda create -n unitab python=3.8
conda activate unitab
```

Install packages in ``requirements.txt`` (separately install [numpy](https://pypi.org/project/numpy/) and [pytorch (LTS 1.8.2)](https://pytorch.org/get-started/locally/) if fails):
```
pip install -r requirements.txt
```

### AzCopy
We recommend using the following AzCopy command to download.
AzCopy executable tools can be [downloaded here](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy).

Example command:
```
path/to/azcopy copy <folder-link> <target-address> --resursive"

# For example:
path/to/azcopy copy https://unitab.blob.core.windows.net/data/data <local_path> --recursive
path/to/azcopy copy https://unitab.blob.core.windows.net/data/weights <local_path> --recursive
path/to/azcopy copy https://unitab.blob.core.windows.net/data/annotations <local_path> --recursive
```

### Distributed Training
We do not specify ``distributed training`` tool in the example commands below. Pytorch distributed ``python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py`` or [submitit](https://github.com/facebookincubator/submitit) supported. Or update ``util/dist.py/init_distributed_mode()`` to fit your cluster setting.


## Data

* Download the original Flickr30k image dataset from : [Flickr30K webpage](http://shannon.cs.illinois.edu/DenotationGraph/) and update the `flickr_img_path` to the folder containing the images.
* Download the original Flickr30k entities annotations from: [Flickr30k annotations](https://github.com/BryanPlummer/flickr30k_entities) and update the `flickr_dataset_path` to the folder with annotations.
* Download the gqa images at [GQA images](https://nlp.stanford.edu/data/gqa/images.zip) and update `vg_img_path` to point to the folder containing the images.
* Download COCO images [Coco train2014](http://images.cocodataset.org/zips/train2014.zip). Update the `coco_path` to the folder containing the downloaded images.

Or download the [cached data (~77G)](https://unitab.blob.core.windows.net/data/data) (use AzCopy with the link).

* Download our pre-processed [annotations (~3.7G)](https://unitab.blob.core.windows.net/data/annotations) (use AzCopy with the link, or [zip file](https://unitab.blob.core.windows.net/data/annotations.zip)) and update the `flickr_ann_path`, `gqa_ann_path` and `refexp_ann_path` to this folder with pre-processed annotations.

## Pre-train
The config file for pretraining is ``configs/pretrain.json``. Optionally starting from [MDETR](https://github.com/ashkamath/mdetr/blob/main/.github/pretrain.md) pretrain with ``--load https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth``. [Weights availble here](https://unitab.blob.core.windows.net/data/weights/pretrained_checkpoint.pth).

Example command (ngpu=64):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/pretrain.json --batch_size 2 --lr_backbone 2e-5 --text_encoder_lr 2e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --unitab_pretrain --pretrain_seqcrop mixed --ema --output-dir weights/$exp_id --load https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth
```

## Multi-task Finetuning
The config file for pretraining is ``configs/multitask.json``. [Weights availble here](https://unitab.blob.core.windows.net/data/weights/prefinetune_checkpoint.pth).

Example command (ngpu=32):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/multitask.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 1e-5 --lr 5e-5 --num_queries 200 --max_decoding_step 256 --load weights/pretrained_checkpoint.pth --ema --output-dir weights/$exp_id
```

## Downstream tasks
Optionally, downloading all weights at once (~54G):
```
path/to/azcopy copy https://unitab.blob.core.windows.net/data/weights <local_path> --recursive
```

For model inference, use the input arguments ``--eval --test``. For captioning tests (Flickr grounded captioning, COCO image captioning, VQAv2 visual question answering), the computed captioning metrics displayed is only for reference. For the final number, an output prediction json file will be automatically stored at ``weights/$exp_id/results/pred_dict_$CIDEr.json``. Please follow the official evaluation for [Flickr grounded captioning](https://github.com/facebookresearch/grounded-video-description), [COCO captioning](https://github.com/tylin/coco-caption), and [VQAv2](https://visualqa.org/evaluation.html) evaluation. We will better intergrate the caption evaluations in future versions.

### Grounded captioning
The config file for pretraining is ``configs/flickr_kp.json``. For model inference, use the input arguments ``--eval --test``. 

For the final number, an output prediction json file will be automatically stored at ``weights/$exp_id/results/pred_dict_$CIDEr.json``. Please follow the official evaluation for [Flickr grounded captioning](https://github.com/facebookresearch/grounded-video-description) evaluation. We will better intergrate the caption evaluations in future versions.

Weights: [Separate](https://unitab.blob.core.windows.net/data/weights/separate_flickrcaptionKP_checkpoint.pth), [Pre-finetuning](https://unitab.blob.core.windows.net/data/weights/prefinetune_flickrcaptionKP_checkpoint.pth).

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>CIDEr</th>
            <th>F1_all</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Separate</td>
            <td>65.6</td>
            <td>11.46</td>
        </tr>
        <tr>
            <td>Pre-finetuning</td>
            <td>69.7</td>
            <td>12.95 </td>
        </tr>
    </tbody>
</table>

Example command (ngpu=8):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr_kp.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 1e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --ema --output-dir weights/$exp_id --load weights/pretrained_checkpoint.pth

CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr_kp.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 1e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --ema --output-dir weights/$exp_id --load weights/prefinetune_flickrcaptionKP_checkpoint.pth --eval --test
```

### Referring expression comprehension
The config file for pretraining is ``configs/refcoco/+/g.json``. For model inference, use the input arguments ``--eval --test --test_type testA/testB/test``.

Weights: [Separate](https://unitab.blob.core.windows.net/data/weights/separate_refcoco_checkpoint.pth), [Pre-finetuning](https://unitab.blob.core.windows.net/data/weights/prefinetune_refcoco_checkpoint.pth) (refcoco/refcoco+/refcocog).

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Refcoco</th>
            <th>Refcoco+</th>
            <th>Refcocog</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Separate</td>
            <td>86.32</td>
            <td>78.70</td>
            <td>79.96</td>
        </tr>
        <tr>
            <td>Pre-finetuning</td>
            <td>88.59</td>
            <td>80.97</td>
            <td>84.58</td>
        </tr>
    </tbody>
</table>

Example command (ngpu=8):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/refcoco.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 5e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --ema --output-dir weights/$exp_id --load weights/pretrained_checkpoint.pth

CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/refcoco.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 5e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --ema --output-dir weights/$exp_id --load weights/prefinetune_refcoco_checkpoint.pth --eval --test --test_type testA
```

### Phrase grounding
The config file for pretraining is ``configs/flickr.json``. For model inference, use the input arguments ``--eval --test``.

Weights: [Separate](https://unitab.blob.core.windows.net/data/weights/separate_flickrGrounding_checkpoint.pth), [Pre-finetuning](https://unitab.blob.core.windows.net/data/weights/prefinetune_flickrGrounding_checkpoint.pth).

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Flickr</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Separate</td>
            <td>79.39</td>
        </tr>
        <tr>
            <td>Pre-finetuning</td>
            <td>79.58</td>
        </tr>
    </tbody>
</table>

Example command (ngpu=8):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 5e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --ema --do_flickrgrounding --output-dir weights/$exp_id --load weights/pretrained_checkpoint.pth

CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr.json --batch_size 2 --lr_backbone 1e-5 --text_encoder_lr 5e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --ema --do_flickrgrounding --output-dir weights/$exp_id --load weights/prefinetune_flickrGrounding_checkpoint.pth --eval --test
```

### COCO captioning
The config file for pretraining is ``configs/flickr_cococaption.json``. For model inference, use the input arguments ``--eval --test``. 

For the final number, an output prediction json file will be automatically stored at ``weights/$exp_id/results/pred_dict_$CIDEr.json``. Please follow the official evaluation for [COCO captioning](https://github.com/tylin/coco-caption) evaluation. We will better intergrate the caption evaluations in future versions.

Weights: [Separate](https://unitab.blob.core.windows.net/data/weights/separate_MScococaption_checkpoint.pth), [Pre-finetuning](https://unitab.blob.core.windows.net/data/weights/prefinetune_MScococaption_checkpoint.pth).

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>CIDEr</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Separate</td>
            <td>119.3</td>
        </tr>
        <tr>
            <td>Pre-finetuning</td>
            <td>119.8</td>
        </tr>
    </tbody>
</table>

Example command (ngpu=16):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr_cococaption.json --lr_backbone 2e-5 --text_encoder_lr 2e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --ema --output-dir weights/$exp_id --load weights/pretrained_checkpoint.pth

CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr_cococaption.json --lr_backbone 2e-5 --text_encoder_lr 2e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --ema --output-dir weights/$exp_id --load weights/prefinetune_MScococaption_checkpoint.pth --eval --test
```

### Visual question answering on VQAv2
The config file for pretraining is ``configs/flickr_vqav2caption.json`` and ``configs/flickr_vqav2captionKP.json``. Adjust the ``GT_type`` between ``vqav2caption`` and ``vqav2captionKP`` for std and KP splits. For model inference, use the input arguments ``--eval --test``. 

For the final number, an output prediction json file will be automatically stored at ``weights/$exp_id/results/pred_dict_$CIDEr.json``. Please follow the official evaluation for [VQAv2](https://visualqa.org/evaluation.html) evaluation. We will better intergrate the caption evaluations in future versions.

Weights: [Separate](https://unitab.blob.core.windows.net/data/weights/separate_VQAv2_checkpoint.pth), [Pre-finetuning](https://unitab.blob.core.windows.net/data/weights/prefinetune_VQAv2_checkpoint.pth). KP split: [Separate](https://unitab.blob.core.windows.net/data/weights/separate_VQAv2KP_checkpoint.pth), [Pre-finetuning](https://unitab.blob.core.windows.net/data/weights/prefinetune_VQAv2KP_checkpoint.pth).


<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>test-dev</th>
            <th>KP-test</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Separate</td>
            <td>69.9</td>
            <td>66.6</td>
        </tr>
        <tr>
            <td>Pre-finetuning</td>
            <td>70.7</td>
            <td>67.5</td>
        </tr>
    </tbody>
</table>

Example command (ngpu=16):
```
CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr_vqav2caption.json --lr_backbone 2e-5 --text_encoder_lr 2e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --ema --output-dir weights/$exp_id --load weights/pretrained_checkpoint.pth

CUBLAS_WORKSPACE_CONFIG=:4096:8  python main.py --dataset_config configs/flickr_vqav2caption.json --lr_backbone 2e-5 --text_encoder_lr 2e-5 --lr 1e-4 --num_queries 200 --max_decoding_step 256 --do_caption --no_detection --ema --output-dir weights/$exp_id --load weights/prefinetune_VQAv2_checkpoint.pth --eval --test
```

## Acknowledgement
The project is built based on the following repository:
* [MDETR--Modulated Detection for End-to-End Multi-Modal Understanding](https://github.com/ashkamath/mdetr),
* [transformers](https://github.com/huggingface/transformers).

### Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
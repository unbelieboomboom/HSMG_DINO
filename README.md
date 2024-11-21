<<<<<<< HEAD
# HSMG_DINO
HSMGDINO is a vision-language multimodal detector designed for environmental perception in automated driving systems.
=======

## Introduction

HSMGDINO is a vision-language multimodal detector designed for environmental perception
in automated driving systems. Its primary innovation lies in the adaptive hard sample
mining mechanism, which leverages contrastive learning to enhance its performance.

## Installation

HSMGDINO is built on [MMetection](https://github.com/open-mmlab/mmdetection).

Please refer to [Installation](https://github.com/open-mmlab/mmdetection/docs/en/get_started.md/#Installation) for installation instructions.

## Getting Started

Please see [get_started.md](https://github.com/open-mmlab/mmdetection/docs/en/get_started.md) for the basic usage of MMDetection.


## Training PFG DINO:

### Preparation:
Chang the config:

```shell
configs/HSMG_DINO/cityscapes_local_detection.py
```
and set 
```shell
data_root = 'Path_to_your_Cityscapes_dataset'
```
Provided here is a download link for the annotation files (including the train, val, 
and test sets) of the Cityscapes dataset, which are formatted in the MS COCO style.

[Download Link (code: pmki)](https://pan.baidu.com/s/1SP_9UFXf41gS5c37tUZ_5w?pwd=pmki)

### Start the training

```shell
python tools/train.py \
    configs/HSMG_DINO/hsm_grounding_dino_hsmdecoderloss_swin-t_finetune_b2_12e_cityscapes.py 
```

## Model Zoo

dataset: Cityscaspes

| Model            | Backbone   | mAP(%) | AR(%) | Download                                                               |
|------------------|------------|--------|-------|------------------------------------------------------------------------|
| 'YOLOv8-m'       | CSPDarkNet | 41.2   | 56.6  |                                                                        |
| `Grounding DINO` | Swin-T     | 48.9   | 67.1  |                                                                        |
| `PFG DINO`       | Swin-T     | 50.7   | 67.9  | [code: 21b5](https://pan.baidu.com/s/11DA4fdWHg4KRfVGmLqk9aA?pwd=21b5) |


dataset: MS COCO 2017

| Model            | Backbone   | mAP(%) | Download                                                               |
|------------------|------------|--------|------------------------------------------------------------------------|
| 'YOLOv8-x'       | CSPDarkNet | 52.7   |                                                                        |
| `Grounding DINO` | Swin-T     | 57.3   |                                                                        |
| `PFG DINO`       | Swin-T     | 57.8   | [code: 566g](https://pan.baidu.com/s/1Dnuk_FN0CPbPqdX1V8_mIg?pwd=566g) |


## Warning
The code is being refactored to ensure standardization, and both the code and the trained
models (which may contain some bugs) will continue to be updated.
>>>>>>> 531ad9f (First submit)

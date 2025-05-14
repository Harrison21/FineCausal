# FineCausal: A Causal-Based Framework for Interpretable Fine-Grained Action Quality Assessment

This repository contains the PyTorch implementation for FineCausal.
[[Paper]](https://arxiv.org/pdf/2503.23911)

## Requirements

Make sure the following dependencies installed (python):

* pytorch >= 0.4.0
* matplotlib=3.1.0
* einops
* timm
* torch_videovision

```
pip install git+https://github.com/hassony2/torch_videovision
```


## Dataset & Annotations

### FineDiving-HM Download

To download the FineDiving-HM dataset and annotations, please follow [FineParser](https://github.com/PKU-ICST-MIPL/FineParser_CVPR2024).

### Data Structure

```
$DATASET_ROOT
├── FineDiving
|  ├── FINADivingWorldCup2021_Men3m_final_r1
|     ├── 0
|        ├── 00489.jpg
|        ...
|        └── 00592.jpg
|     ...
|     └── 11
|        ├── 14425.jpg
|        ...
|        └── 14542.jpg
|  ...
|  └── FullMenSynchronised10mPlatform_Tokyo2020Replays_2
|     ├── 0
|     ...
|     └── 16 
└──FineDiving_HM
|  ├── FINADivingWorldCup2021_Men3m_final_r1
|     ├── 0
|        ├── 00489.jpg
|        ...
|        └── 00592.jpg
|     ...
|     └── 11
|        ├── 14425.jpg
|        ...
|        └── 14542.jpg
|  ...
|  └── FullMenSynchronised10mPlatform_Tokyo2020Replays_2
|     ├── 0
|     ...
|     └── 16 

$ANNOTATIONS_ROOT
|  ├── FineDiving_coarse_annotation.pkl
|  ├── FineDiving_fine-grained_annotation.pkl
|  ├── Sub_action_Types_Table.pkl
|  ├── fine-grained_annotation_aqa.pkl
|  ├── train_split.pkl
|  ├── test_split.pkl
```

## Training
To download pretrained_i3d_wight, please follow [kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch/tree/master), and put `model_rgb.pth` in `models` folder.

To train the model, please run:
```bash
python launch.py
```

## Test
To test the trained model, please set `test: True` in [config](FineDiving_FineParser.yaml) and run:
```bash
python launch.py
```

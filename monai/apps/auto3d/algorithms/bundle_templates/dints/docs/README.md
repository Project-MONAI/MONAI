# Description
A neural architecture search algorithm for volumetric (3D) segmentation of the pancreas and pancreatic tumor from CT image.

# Model Overview
This model is trained using the state-of-the-art algorithm [1] of the "Medical Segmentation Decathlon Challenge 2018" with 196 training images, 56 validation images, and 28 testing images.

## Data
The training dataset is Task07_Pancreas.tar from http://medicaldecathlon.com/. And the data list/split can be created with the script `scripts/prepare_datalist.py`.

## Training configuration
The training was performed with at least 16GB-memory GPUs.

Actual Model Input: 96 x 96 x 96

## Input and output formats
Input: 1 channel CT image

Output: 3 channels: Label 2: pancreatic tumor; Label 1: pancreas; Label 0: everything else

## Scores
This model achieves the following Dice score on the validation data (our own split from the training dataset):

Mean Dice = 0.72

## commands example
Create data split (.json file):

```
python scripts/prepare_datalist.py --path /path-to-Task07_Pancreas/ --output configs/dataset_0.json
```

Execute model searching:

```
python -m scripts.search run --config_file configs/search.yaml
```

Execute multi-GPU model searching (recommended):

```
torchrun --nnodes=1 --nproc_per_node=8 -m scripts.search run --config_file configs/search.yaml
```

Execute training:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.yaml --logging_file configs/logging.conf
```

Override the `train` config to execute multi-GPU training:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run training --meta_file configs/metadata.json --config_file "['configs/train.yaml','configs/multi_gpu_train.yaml']" --logging_file configs/logging.conf
```

Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.yaml','configs/evaluate.yaml']" --logging_file configs/logging.conf
```

Execute inference:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.yaml --logging_file configs/logging.conf
```

# Disclaimer
This is an example, not to be used for diagnostic purposes.

# References
[1] He, Y., Yang, D., Roth, H., Zhao, C. and Xu, D., 2021. Dints: Differentiable neural network topology search for 3d medical image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 5841-5850).

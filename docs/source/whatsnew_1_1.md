# What's new in 1.1 🎉🎉

- Experiment management for MONAI bundle
- New models in MONAI Model Zoo
- State-of-the-art SurgToolLoc solution

## Experiment management for MONAI bundle
![exp_mgmt](../images/exp_mgmt.png)

In this release, experiment management features are integrated with the MONAI bundle.
It provides essential APIs for managing the end-to-end model bundle lifecycle.
Users can start tracking experiments by, for example, appending `--tracking "mlflow"` to the training or inference bundle commands to enable the MLFlow-based management. 
For more details about it, please refer to this [tutorial](https://github.com/Project-MONAI/tutorials/blob/main/experiment_management/bundle_integrate_mlflow.ipynb).

## New models in MONAI Model Zoo
New pretrained models are being created and released [in the Model Zoo](https://monai.io/model-zoo.html).
Notably,
- The `mednist_reg` model demonstrate how to build image registration workflow using 
a ResNet and spatial transformer for hand X-ray images in MONAI bundle
format (based on [the registration_mednist tutorial](https://github.com/Project-MONAI/tutorials/blob/main/2d_registration/registration_mednist.ipynb)).

For more details about how to use the models, please see [the tutorials](https://github.com/Project-MONAI/tutorials/tree/main/model_zoo).

## State-of-the-art SurgToolLoc solution
[SurgToolLoc](https://surgtoolloc.grand-challenge.org/Home/) is a part of
[EndoVis](https://endovis.grand-challenge.org/) challenge at [MICCAI 2022](https://conferences.miccai.org/2022/en/).
The challenge focuses on endoscopic video analysis and is divided into (1) fully supervised tool classification
and (2) weakly supervised tool classification/localization.
Team NVIDIA won prizes by finishing [third](https://surgtoolloc.grand-challenge.org/results/) in both categories.
The core components of the solutions are released in MONAI. For more details about the implementation,
please see [the tutorials](https://github.com/Project-MONAI/tutorials/tree/main/competitions/MICCAI/surgtoolloc).

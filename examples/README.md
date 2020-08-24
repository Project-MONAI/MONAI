### 1. Requirements
Some of the examples may require optional dependencies. In case of any optional import errors,
please install the relevant packages according to the error message.
Or install all optional requirements by:
```
pip install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/master/requirements-dev.txt
```

### 2. List of examples
#### [classification_3d](./classification_3d)
Training and evaluation examples of 3D classification based on DenseNet3D and [IXI dataset](https://brain-development.org/ixi-dataset).
The examples are standard PyTorch programs and have both dictionary-based and array-based transformation versions.
#### [classification_3d_ignite](./classification_3d_ignite)
Training and evaluation examples of 3D classification based on DenseNet3D and [IXI dataset](https://brain-development.org/ixi-dataset).
The examples are PyTorch Ignite programs and have both dictionary-based and array-based transformation versions.
#### [distributed_training](./distributed_training)
The examples show how to execute distributed training and evaluation based on 3 different frameworks:
- PyTorch native `DistributedDataParallel` module with `torch.distributed.launch`.
- Horovod APIs with `horovodrun`.
- PyTorch ignite and MONAI workflows.

They can run on several distributed nodes with multiple GPU devices on every node.
#### [segmentation_3d](./segmentation_3d)
Training and evaluation examples of 3D segmentation based on UNet3D and synthetic dataset.
The examples are standard PyTorch programs and have both dictionary-based and array-based versions.
#### [segmentation_3d_ignite](./segmentation_3d_ignite)
Training and evaluation examples of 3D segmentation based on UNet3D and synthetic dataset.
The examples are PyTorch Ignite programs and have both dictionary-base and array-based transformations.
#### [workflows](./workflows)
Training and evaluation examples of 3D segmentation based on UNet3D and synthetic dataset.
The examples are built with MONAI workflows, mainly contain: trainer/evaluator, handlers, post_transforms, etc.
#### [synthesis](./synthesis)
A GAN training and evaluation example for a medical image generative adversarial network. Easy run training script uses `GanTrainer` to train a 2D CT scan reconstruction network. Evaluation script generates random samples from a trained network.

### 3. List of tutorials
Please check out https://github.com/Project-MONAI/Tutorials

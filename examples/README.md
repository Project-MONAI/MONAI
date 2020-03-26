### 1. Requirements
Most of the examples and tutorials require
[matplotlib](https://matplotlib.org/) and [Jupyter Notebook](https://jupyter.org/).

These could be installed by:
```bash
python -m pip install -U pip
python -m pip install -U matplotlib
python -m pip install -U notebook
```

### 2. List of examples
#### 1. [classification_3d](https://github.com/Project-MONAI/MONAI/tree/master/examples/classification_3d)
Training and evaluation examples of 3D classification based on DenseNet3D and IXI dataset:  
https://brain-development.org/ixi-dataset  
The examples are standard PyTorch programs and have both `dictionary data` and `array data` versions.
#### 2. [classification_3d_ignite](https://github.com/Project-MONAI/MONAI/tree/master/examples/classification_3d_ignite)
Training and evaluation examples of 3D classification based on DenseNet3D and IXI dataset:  
https://brain-development.org/ixi-dataset  
The examples are PyTorch ignite programs and have both `dictionary data` and `array data` versions.
#### 3. [notebooks/multi_gpu_test](https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/multi_gpu_test.ipynb)
This notebook is a quick test for devices, run the ignite trainer engine on CPU, single GPU and multi-GPUs.
#### 4. [notebooks/nifti_read_example](https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/nifti_read_example.ipynb)
Illustrate reading Nifti files and iterating over patches of the volumes loaded from them.
#### 5. [notebooks/spleen_segmentation_3d](https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/spleen_segmentation_3d.ipynb)
This notebook is an end-to-end training & evaluation example of 3D segmentation based on MSD Spleen dataset:  
http://medicaldecathlon.com  
The example is a standard PyTorch program and shows several key features of MONAI:  
(1) Transforms for dictionary format data.  
(2) Load Nifti image with metadata.  
(3) Scale medical image intensity with expected range.  
(4) Crop out a batch of balanced images based on positive / negative label ratio.  
(5) 3D UNet model, Dice loss function, Mean Dice metric for 3D segmentation task.  
(6) Sliding window inference method.  
(7) Deterministic training for reproducibility.
#### 6. [notebooks/transform_speed](https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/transform_speed.ipynb)
Illustrate reading Nifti files and test speed of different transforms on different devices.
#### 7. [notebooks/transforms_demo_2d](https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/transforms_demo_2d.ipynb)
This notebook demonstrates the medical domain specific transforms on 2D medical images.
#### 8. [notebooks/unet_segmentation_3d_ignite](https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/unet_segmentation_3d_ignite.ipynb)
This notebook is an end-to-end training & evaluation example of 3D segmentation based on synthetic dataset.  
The example is a PyTorch ignite program and shows several key features of MONAI,  
espeically medical domain specific transforms and ignite Event-Handlers.
#### 9. [segmentation_3d](https://github.com/Project-MONAI/MONAI/tree/master/examples/segmentation_3d)
Training and evaluation examples of 3D segmentation based on UNet3D and synthetic dataset.  
The examples are standard PyTorch programs and have both `dictionary data` and `array data` versions.
#### 10. [segmentation_3d_ignite](https://github.com/Project-MONAI/MONAI/tree/master/examples/segmentation_3d_ignite)
Training and evaluation examples of 3D segmentation based on UNet3D and synthetic dataset.  
The examples are PyTorch ignite programs and have both `dictionary data` and `array data` versions.

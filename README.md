# Project MONAI
**M**edical **O**pen **N**etwork for **AI** - _Toolkit for Healthcare Imaging_

_Contact: <monai.miccai2019@gmail.com>_

This document identifies key concepts of project MONAI at a high level, the goal is to facilitate further technical discussions of requirements,roadmap, feasibility and trade-offs.

## Vision
   *   Develop a community of academic, industrial and clinical researchers collaborating and working on a common foundation of standardized tools.
   *   Create a state-of-the-art, end-to-end training toolkit for healthcare imaging.
   *   Provide academic and industrial researchers with the optimized and standardized way to create and evaluate models

## Targeted users
   *   Primarily focused on the healthcare researchers who develop DL models for medical imaging

## Goals
   *   Deliver domain-specific workflow capabilities
   *   Address the end-end “Pain points” when creating medical imaging deep learning workflows.
   *   Provide a robust foundation with a performance optimized system software stack that allows researchers to focus on the research and not worry about software development principles.

## Guiding principles
### Modularity
   *   Pythonic -- object oriented components
   *   Compositional -- can combine components to create workflows
   *   Extensible -- easy to create new components and extend existing components
   *   Easy to debug -- loosely coupled, easy to follow code (e.g. in eager or graph mode)
   *   Flexible -- interfaces for easy integration of external modules
### User friendly
   *   Portable -- use components/workflows via Python “import”
   *   Run well-known baseline workflows in a few commands
   *   Access to the well-known public datasets in a few lines of code
### Standardisation
   *   Unified/consistent component APIs with documentation specifications
   *   Unified/consistent data and model formats, compatible with other existing standards
### High quality
   *   Consistent coding style - extensive documentation - tutorials - contributors’ guidelines
   *   Reproducibility -- e.g. system-specific deterministic training
### Future proof
   *   Task scalability -- both in datasets and computational resources
   *   Support for advanced data structures -- e.g. graphs/structured text documents
### Leverage existing high-quality software packages whenever possible
   *   E.g. low-level medical image format reader, image preprocessing with external packages
   *   Rigorous risk analysis of choice of foundational software dependencies
### Compatible with external software
   *   E.g. data visualisation, experiments tracking, management, orchestration

## Key capabilities

<table>
  <tr>
   <td>
<strong><em>Basic features</em></strong>
   </td>
   <td colspan="2" ><em>Example</em>
   </td>
   <td><em>Notes</em>
   </td>
  </tr>
  <tr>
   <td>Ready-to-use workflows
   </td>
   <td colspan="2" >Volumetric image segmentation
   </td>
   <td>“Bring your own dataset”
   </td>
  </tr>
  <tr>
   <td>Baseline/reference network architectures
   </td>
   <td colspan="2" >Provide an option to use “U-Net”
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Intuitive command-line interfaces
   </td>
   <td colspan="2" >
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Multi-gpu training
   </td>
   <td colspan="2" >Configure the workflow to run data parallel training
   </td>
   <td>
   </td>
  </tr>
</table>



<table>
  <tr>
   <td><strong><em>Customisable Python interfaces</em></strong>
   </td>
   <td colspan="2" ><em>Example</em>
   </td>
   <td><em>Notes</em>
   </td>
  </tr>
  <tr>
   <td>Training/validation strategies
   </td>
   <td colspan="2" >Schedule a strategy of alternating between generator and discriminator model training
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Network architectures
   </td>
   <td colspan="2" >Define new networks w/ the recent “Squeeze-and-Excitation” blocks
   </td>
   <td>“Bring your own model”
   </td>
  </tr>
  <tr>
   <td>Data preprocessors
   </td>
   <td colspan="2" >Define a new reader to read training data from a database system
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Adaptive training schedule
   </td>
   <td colspan="2" >Stop training when the loss becomes “NaN”
   </td>
   <td>“Callbacks”
   </td>
  </tr>
  <tr>
   <td>Configuration-driven workflow assembly
   </td>
   <td colspan="2" >Making workflow instances from configuration file
   </td>
   <td>Convenient for managing hyperparameters
   </td>
  </tr>
</table>



<table>
  <tr>
   <td><strong><em>Model sharing & transfer learning</em></strong>
   </td>
   <td colspan="2" ><em>Example</em>
   </td>
   <td><em>Notes</em>
   </td>
  </tr>
  <tr>
   <td>Sharing model parameters, hyperparameter configurations
   </td>
   <td colspan="2" >Standardisation of model archiving format
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Model optimisation for deployment
   </td>
   <td colspan="2" >
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Fine-tuning from pre-trained models
   </td>
   <td colspan="2" >Model compression, TensorRT
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Model interpretability
   </td>
   <td colspan="2" >Visualising feature maps of a trained model
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Experiment tracking & management
   </td>
   <td colspan="2" >
   </td>
   <td><a href="https://polyaxon.com/">https://polyaxon.com/</a>
   </td>
  </tr>
</table>



<table>
  <tr>
   <td><strong><em>Advanced features</em></strong>
   </td>
   <td colspan="2" ><em>Example</em>
   </td>
   <td><em>Notes</em>
   </td>
  </tr>
  <tr>
   <td>Compatibility with external toolkits
   </td>
   <td colspan="2" >XNAT as data source, ITK as preprocessor
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Advanced learning strategies
   </td>
   <td colspan="2" >Semi-supervised, active learning
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>High performance preprocessors
   </td>
   <td colspan="2" >Smart caching, multi-process
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Multi-node distributed training
   </td>
   <td colspan="2" >
   </td>
   <td>
   </td>
  </tr>
</table>

## Highlight features
As a medical area-specific Deep Learning package, MONAI provides rich support on end-to-end medical DL research work, for example, 2D/3D medical image processing, model training & evaluation(like segmentation, classification tasks), statistics & 2D/3D image visualization, etc.  
Here are the highlight features:
### 1. Transforms
Data pre-processing is an important and challenging task in medical DL research work because the images are in the medical specific format with rich metadata information. And usually, the images are very large and in 3D shape which is much different from other common computer vision areas. So MONAI provides powerful and flexible transforms to enable these features:
#### 1. Transforms support both Dictionary and Array format data
(1) The traditional transforms(like Torchvision) execute operations on array data directly(rotate image, etc.), MONAI provides more transforms for both 2D and 3D data and can be composed with Torchvision transforms together.  
(2) Additionally, as medical images contain much more information than common images and many DL tasks need to transform several data together(randomly crop image and label together in segmentation, etc.), MONAI also provides transforms for dictionary format 2D/3D data. Users just need to specify the keys corresponding to the expected data fields to operate transforms.  
#### 2. Medical specific transforms
MONAI provides many transforms that are popular in the medical DL area but not supported in common PyTorch libraries.  
For example(check the webpage for all the transforms):  
| Name | Description |
| ------ | ------ |
| LoadNifti | Load Nifti format file from provided path |
| Spacing | Resample input image into the specified `pixdim` |
| Orientation | Change image's orientation into the specified `axcodes` |
| IntensityNormalizer | Normalize image based on mean and std |
| AffineGrid | Affine transforms on the coordinates |
| RandDeformGrid | Generate random deformation grid |
| Affine | Transform image based on the affine parameters |
| Rand3DElastic | Random elastic deformation and affine in 3D |
#### 3. Fused spatial transforms and GPU acceleration
As medical images are usually larger(especially 3D), pre-processing performance obviously affects the DL research work. So MONAI provides several affine transforms to execute fused spatial operations. And even support GPU acceleration to achieve much faster performance.  
Example code:
```py
# create an Affine transform
affine = Affine(
    rotate_params=np.pi/4,
    scale_params=(1.2, 1.2),
    translate_params=(200, 40), 
    padding_mode='zeros',
    device=torch.device('cuda:0')
)
# convert the image using interpolation mode
new_img = affine(image, spatial_size=(300, 400), mode='bilinear')
```
#### 4. Randomly crop out batch images based on positive/negative ratio
Many medical images are very large and the dataset doesn't contain many labelled samples for training. A useful method is to crop out a batch of smaller samples from a big image. And the optimized method is to crop based on positive/negative labels ratio which can provide more stable and balanced dataset for the training process.  
MONAI provides a transform "RandCropByPosNegLabelDict" for this feature, and all the following transforms in the chain can seamlessly operate on every item in the cropped batch.  
### 2. Losses
There are many specific and useful loss functions in the medical DL research area which are different from common PyTorch loss functions. That's an important module of MONAI, and already implemented Dice related loss functions,  will integrate more loss functions soon.  
### 3. Models
DL method in medical research work is becoming more and more popular in recent years and many networks come up to show great improvement in tasks like classification and segmentation, etc. MONAI already implemented UNet and DenseNet to support both 2D/3D segmentation and classification tasks, will integrate more networks soon.  
### 4. Evaluation
In order to accurately evaluate the model and execute inference, there are many optimized methods and metrics, MONAI already implemented several popular ones of them:
#### 1. Sliding window inference
When executing inference on very big medical images, the sliding window is a very popular method to achieve high accuracy without much memory.  
(1) Select continuous windows on the original image.  
(2) Execute a batch of windows on the model per time, and complete all windows.  
(3) Connect all the model outputs to construct 1 segmentation corresponding to the original image.  
(4) Save segmentation result to file or compute metrics.  

#### 2. Metrics for medical tasks
There are many useful metrics to measure medical specific tasks, MONAI already implemented Mean Dice and AUC, will integrate more soon.  
### 5. Visualization
Besides common curves of statistics on TensorBoard, in order to provide straight-forward checking of 3D image and the corresponding label & segmentation output, MONAI can draw a 3D data as GIF image on TensorBoard which can help users check model output very quickly.
### 6. Savers
For the segmentation task, MONAI can support to save model output as NIFTI format image and add affine information from corresponding input image.  
For the classification task, MONAI can support to save classification result as a CSV sheet.

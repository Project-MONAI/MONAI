# Modules in v0.2.0

MONAI aims at supporting deep learning in medical image analysis at multiple granularities.
This figure shows a typical example of the end-to-end workflow in medical deep learning area:
![image](../images/end_to_end.png)

## MONAI architecture
The design principle of MONAI is to provide flexible and light APIs for different users.
1. All the core components are independent modules, which can be easily integrated into any existing PyTorch programs to help researchers on specific functions.
2. On the other hand, users can leverage the workflows in MONAI to quickly set up a robust training or evaluation program for experiments.
3. MONAI provides rich examples and notebooks to demonstrate core features.
4. Researchers also contributed many implementations corresponding to state-of-the-art papers, including latest research challenges like: COVID-19 analysis and Model Parallel, Federated Learning, etc.

The overall architecture and modules are showed in below figure:
![image](../images/arch_modules_v0.2.png)
The rest of this page provides more details for each module.

* [Data I/O, processing and augmentation](#medical-image-data-io-processing-and-augmentation)
* [Datasets](#datasets)
* [Loss functions](#losses)
* [Network architectures](#network-architectures)
* [Evaluation](#evaluation)
* [Visualization](#visualization)
* [Result writing](#result-writing)
* [Workflows](#workflows)
* [Research](#research)

## Medical image data I/O, processing and augmentation
Medical images require highly specialized methods for I/O, preprocessing, and augmentation. Medical images are often in specialized formats with rich meta-information, and the data volumes are often high-dimensional. These require carefully designed manipulation procedures. The medical imaging focus of MONAI is enabled by powerful and flexible image transformations that facilitate user-friendly, reproducible, optimized medical data pre-processing pipelines.

### 1. Transforms support both Dictionary and Array format data
- The widely used computer vision packages (such as ``torchvision``) focus on spatially 2D array image processing. MONAI provides more domain-specific transformations for both spatially 2D and 3D and retains the flexible transformation "compose" feature.
- As medical image preprocessing often requires additional fine-grained system parameters, MONAI provides transforms for input data encapsulated in python dictionaries. Users can specify the keys corresponding to the expected data fields and system parameters to compose complex transformations.

There are huge number of transforms in 6 categories: Crop & Pad, Intensity, IO, Post, Spatial and Utilities. For more details, please check: [all the transforms in MONAI](https://monai.readthedocs.io/en/latest/transforms.html).

### 2. Medical specific transforms
MONAI aims at providing a rich set of popular medical image specific
transformations. These currently include, for example:
- `LoadNifti`:  Load Nifti format file from provided path
- `Spacing`:  Resample input image into the specified `pixdim`
- `Orientation`: Change the image's orientation into the specified `axcodes`
- `RandGaussianNoise`: Perturb image intensities by adding statistical noises
- `NormalizeIntensity`: Intensity Normalization based on mean and standard deviation
- `Affine`: Transform image based on the affine parameters
- `Rand2DElastic`: Random elastic deformation and affine in 2D
- `Rand3DElastic`: Random elastic deformation and affine in 3D

### 3. Fused spatial transforms and GPU acceleration
As medical image volumes are usually large (in multi-dimensional arrays),
pre-processing performance obviously affects the overall pipeline speed. MONAI provides affine transforms to execute fused spatial operations, supports GPU acceleration via native PyTorch to achieve high performance.  
For example:
```py
# create an Affine transform
affine = Affine(
    rotate_params=np.pi/4,
    scale_params=(1.2, 1.2),
    translate_params=(200, 40),
    padding_mode='zeros',
    device=torch.device('cuda:0')
)
# convert the image using bilinear interpolation
new_img = affine(image, spatial_size=(300, 400), mode='bilinear')
```
And all the spatial transforms(Spacing, Zoom, Rotate, Resize, etc.) are designed based on PyTorch native interfaces (instead of scipy, scikit-image, etc.).

### 4. Randomly crop out batch images based on positive/negative ratio
Medical image data volume may be too large to fit into GPU memory. A widely-used approach is to randomly draw small size data samples during training and run a “sliding window” routine for inference.  MONAI currently provides general random sampling strategies including class-balanced fixed ratio sampling which may help stabilize the patch-based training process.

### 5. Deterministic training for reproducibility
Deterministic training support is necessary and important for deep learning research, especially in the medical field. Users can easily set the random seed to all the random transforms in MONAI locally and will not affect other non-deterministic modules in the user's program.
For example:
```py
# define a transform chain for pre-processing
train_transforms = monai.transforms.Compose([
    LoadNiftid(keys=['image', 'label']),
    RandRotate90d(keys=['image', 'label'], prob=0.2, spatial_axes=[0, 2]),
    ... ...
])
# set determinism for reproducibility
train_transforms.set_random_state(seed=0)
```
Users can also enable/disable deterministic at the beginning of training program:
```py
monai.utils.set_determinism(seed=0, additional_settings=None)
```

### 6. Multiple transform chains
In order to execute different transforms on the same data and concat the results or give results to model directly, MONAI provides `CopyItems` transform to make copies of specified items in the data dictionary and `ConcatItems` transform to concat specified items on expected dimension, and also provides `DeleteItems` transform to delete unnecessary items to save memory.  
A typical usage is to scale different intensity range on the same image and concat the results together as a 3 channels data.
![image](../images/multi_transform_chains.png)

### 7. Debug transforms with DataStats
As we usually compose all the transforms in a chain, it's not easy to track the output of specific transform. To help debugging errors in transforms, MONAI provides an utility transform `DataStats` to print information about its input data, like: `data shape`, `intensity range`, `data value` and some `additional info`.  
It's an independent transform and can be integrated into expected place in the transform chain.

### 8. Post transforms for model output
MONAI provides transforms not only for pre-processing but also for post-processing, which are used for many following tasks after model output.  
Currently, these post transforms are available:
- Add activition layer(Sigmoid, Softmax, etc.).
- Convert to discrete values(Argmax, One-Hot, Threshold value, etc).
- Split 1 multi-channel data into different several single channel data.
- Keep only the largest connected component(remove noise in model output).

After the post transforms, it's easier to compute metrics, save model output into files or visualize data in the TensorBoard.

### 9. Integrate 3rd party transforms
The design of MONAI transforms are quite straight forward and work for array data or dictionary data directly. And MONAI also provides `Adaptor` tool to adapt input/output data format for 3rd party transforms. To convert the data shape or data type, users can use the utility transforms `ToTensor`, `ToNumpy`, `SqueeseDim`, etc.  
So it's easy to enhance the transform chain by integrating more transforms from other libraries, like: `ITK`, `BatchGenerator`, `TorchIO` and `Rising`, etc.  
For more details, check the tutorial: [integrate 3rd pary transforms into MONAI program](https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/integrate_3rd_party_transforms.ipynb).

## Datasets
### 1. Cache IO and transforms data to accelerate training
Users often need to train the model with many (potentially thousands of) epochs over the data to achieve the desired model quality. A native PyTorch
implementation may repeatedly load data and run the same preprocessing steps for every epoch during training, which can be time-consuming and unnecessary, especially when the medical image volumes are large.  
MONAI provides a multi-threads `CacheDataset` to accelerate these transformation steps during training by storing the intermediate outcomes before the first randomized transform in the transform chain. Enabling this feature could potentially give 10x training speedups.
![image](../images/cache_dataset.png)

### 2. Cache intermediate outcomes into persistent storage
The `PersistentDataset` is similar to the CacheDataset, where the intermediate cache values are persisted to disk storage for rapid retrieval between experimental runs (as is the case when tuning hyper parameters), or when the entire data set size exceeds available memory.

### 3. Zip several PyTorch datasets and output data together
MONAI provides `ZipDataset` to connect several PyTorch datasets and combine the output data(with the same index) together in a tuple, which can be helpful to execute complicated training progress based on several data sources.  
For example:
```py
class DatasetA(Dataset):
    def __getitem__(self, index: int):
        return image_data[index]

class DatasetB(Dataset):
    def __getitem__(self, index: int):
        return extra_data[index]

dataset = ZipDataset([DatasetA(), DatasetB()], transform)
```

### 4. Predefined Datasets for public medical data
In order to quickly get start with popular training data in medical domain, MONAI provides several data-specific Datasets(like: `MedNISTDataset`, `DecathlonDataset`, etc.), which include downloading, extracting data files and support generation of training/evaluation items with transforms. And they are flexible that users can easily modify the JSON config file to change the default behaviors.  
If anyone wants to contribute a new public dataset, just refer to existing Datasets and leverage the download and extracting APIs, etc.  
The common progress of predefined datasets:
![image](../images/dataset_progress.png)

## Losses
There are domain-specific loss functions in the medical imaging research which are not typically used in the generic computer vision tasks. As an important module of MONAI, these loss functions are implemented in PyTorch, such as `DiceLoss`, `GeneralizedDiceLoss`, `MaskedDiceLoss`, `TverskyLoss` and `FocalLoss`, etc.

## Network architectures
Some deep neural network architectures have shown to be particularly effective for medical imaging analysis tasks. MONAI implements reference networks with the aims of both flexibility and code readability. To leverage the common network layers and blocks, MONAI provides several predefined layers and blocks which are compatible with 1D, 2D and 3D networks. Users can easily integrate the layer factories in their own networks.
For example:
```py
# import MONAI’s layer factory
from monai.networks.layers import Conv

# adds a transposed convolution layer to the network
# which is compatible with different spatial dimensions.
name, dimension = Conv.CONVTRANS, 3
conv_type = Conv[name, dimension]
add_module('conv1', conv_type(in_channels, out_channels, kernel_size=1, bias=False))
```
And there are already several 1D/2D/3D compatible implementation of networks, such as: UNet, DenseNet, GAN, etc.

## Evaluation
To run model inferences and evaluate the model quality, MONAI provides reference implementations for the relevant widely-used approaches. Currently, several popular evaluation metrics and inference patterns are included:

### 1. Sliding window inference
For model inferences on large volumes, the sliding window approach is a popular choice to achieve high performance while having flexible memory requirements. It also supports `overlap` and `blending_mode` parameters to smooth the segmentation result with weights for better metrics.  
The typical progress:
1. Select continuous windows on the original image.
2. Iteratively run batched window inferences until all windows are analyzed.
3. Aggregate the inference outputs to a single segmentation map.
4. Save the results to file or compute some evaluation metrics.
![image](../images/sliding_window.png)

### 2. Metrics for medical tasks
Various useful evaluation metrics have been used to measure the quality of medical image specific models. MONAI already implemented mean Dice score for segmentation tasks and the area under the ROC curve for classification tasks. We continue to integrate more options.

## Visualization
Beyond the simple point and curve plotting, MONAI provides intuitive interfaces to visualize multidimensional data as GIF animations in TensorBoard. This could provide a quick qualitative assessment of the model by visualizing, for example, the volumetric inputs, segmentation maps, and intermediate feature maps.

## Result writing
Currently MONAI supports writing the model outputs as NIfTI files or PNG files for segmentation tasks, and as CSV files for classification tasks. And the writers can restore the data spacing, orientation or shape according to the `original_shape` or `original_affine` information from the input image.  
A rich set of formats will be supported soon, along with relevant statistics and evaluation metrics automatically computed from the outputs.

## Workflows
To quickly set up training/evaluation experiments or pipelines, MONAI provides a set of workflows to significantly simplify the modules, decouple all the components during a training progress, and unify the program APIs for higher level applications, like: AutoML, Federated Learning, etc.  
The trainers and evaluators of the workflows are compatible with ignite `Engine` and `Event-Handler` mechanism. There are rich event handlers in MONAI to independently attach to the trainer or evaluator.  
The workflow progress and event handlers are shown as below:  
![image](../images/workflows.png)

## Research
We encourage deep learning researchers in medical domain to implement their awesome research works based on MONAI components and contribute into MONAI. It can be helpful to cooperate with others for the future research.  
There are already several implementation in MONAI corresponding to latest papers that are trying to solve challenging topics(COVID-19, Model Parallel, etc.), like:
### 1. COPLE-Net for COVID-19 Pneumonia Lesion Segmentation
A reimplementation of the COPLE-Net originally proposed by:  
G. Wang, X. Liu, C. Li, Z. Xu, J. Ruan, H. Zhu, T. Meng, K. Li, N. Huang, S. Zhang. (2020) "A Noise-robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions from CT Images." IEEE Transactions on Medical Imaging. 2020. DOI: 10.1109/TMI.2020.3000314
![image](../images/coplenet.png)

### 2. LAMP: Large Deep Nets with Automated Model Parallelism for Image Segmentation
A reimplementation of the LAMP system originally proposed by:  
Wentao Zhu, Can Zhao, Wenqi Li, Holger Roth, Ziyue Xu, and Daguang Xu (2020) "LAMP: Large Deep Nets with Automated Model Parallelism for Image Segmentation." MICCAI 2020 (Early Accept, paper link: https://arxiv.org/abs/2006.12575)

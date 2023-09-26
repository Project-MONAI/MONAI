# What's new in 1.2

- Auto3DSeg enhancements and benchmarks
- nnUNet integration
- TensorRT-optimized networks
- MetricsReloaded integration
- Bundle workflow APIs
- Modular patch inference
- Lazy resampling for preprocessing

## Auto3DSeg enhancements and benchmarks
Auto3DSeg is an innovative solution for 3D medical image segmentation, leveraging the advancements in MONAI and GPUs for algorithm development and deployment.
Key improvements in this release include:
- Several new modules to the training pipelines, such as automated GPU-based hyperparameter scaling, early stopping mechanisms, and dynamic validation frequency.
- Multi-GPU parallelism has been activated for all GPU-related components including data analysis, model training, and model ensemble, to augment overall performance and capabilities.
- The algorithms were benchmarked for computational efficiency on the TotalSegmentator dataset, containing over 1,000 CT images.
- Multi-node training is implemented, reducing model training time significantly.


## nnUNet integration
The integration introduces a new class, `nnUNetV2Runner`, which leverages Python APIs to facilitate model training, validation,
and ensemble, thereby simplifying the data conversion process for users.
Benchmarking results from various public datasets confirm that nnUNetV2Runner performs as expected.
Users are required to prepare a data list and create an `input.yaml` file to install and use the system.
The framework also allows automatic execution of the entire nnU-Net pipeline, from model training to ensemble,
with options to specify the number of epochs. Users can access APIs for training, dataset conversion, data preprocessing, and other components.
Please check out [the tutorials](https://github.com/Project-MONAI/tutorials/tree/main/nnunet) for more details.

## TensorRT-optimized networks
[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) is an SDK for high-performance deep learning inference,
includes a deep learning inference optimizer and runtime that delivers low latency and high throughput for inference applications.
It can accelerate the deep learning model forward computation on the NVIDIA GPU.
In this release, the `trt_export` API to export the TensorRT engine-based TorchScript model has been integrated into the MONAI bundle.
Users can try to export bundles with it. A few bundles in the MONAI model zoo,
like the [spleen_ct_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/spleen_ct_segmentation)
and [endoscopic_tool_segmentation](https://github.com/Project-MONAI/model-zoo/tree/dev/models/endoscopic_tool_segmentation) bundles,
have already been exported and benchmarked. For more details about how to export and benchmark a model,
please go to this [tutorial](https://github.com/Project-MONAI/tutorials/blob/main/acceleration/TensorRT_inference_acceleration.ipynb).


## MetricsReloaded integration
MetricsReloaded - a new recommendation framework for biomedical image analysis validation - is released publicly
via https://github.com/Project-MONAI/MetricsReloaded. Binary and categorical metrics computing modules are included in this release,
using MetricsReloaded as the backend. [Example scripts](https://github.com/Project-MONAI/tutorials/tree/main/modules/metrics_reloaded) are made available to demonstrate the usage.


## Bundle workflow APIs
`BundleWorkflow` abstracts the typical workflows (such as training, evaluation, and inference) of a bundle with three main interfaces:
`initialize`, `run`, and `finalize`, applications use these APIs to execute a bundle.
It unifies the required properties and optional properties for the workflows, downstream applications
can invoke the components instead of parsing configs with keys.
In this release, `ConfigWorkflow` class is also created for JSON and YAML config-based bundle workflows for improved Pythonic usability.


## Modular patch inference
In patch inference, patches are extracted from the image, the inference is run on those patches, and outputs are merged
to construct the result image corresponding to the input image. Although depending on the task, model, and computational/memory resources,
the exact implementations of a patch inference may vary, the overall process of splitting, running inference, and merging the results remains the same.
In this release, we have created a modular design for patch inference, which defines the overall process while abstracting away the specific
behavior of how to split the image into patches, how to pre and post process each patch, and how to merge the output patches.

## Lazy Resampling for preprocessing
Lazy Resampling is a new, experimental feature for preprocessing. It works under
the hood along with MONAI transforms to combine adjacent spatial and
cropping transforms into a single operation. This allows MONAI to reduce the number of data resamples
 a pipeline undergoes. Depending on the preprocessing pipeline, it can potentially:

* reduce processing time
* reduce processing memory
* reduce incidental artifacts added by resampling
* preserve data that would otherwise be cropped and replaced with padding

Lazy Resampling pipelines can use a mixture of MONAI and non-MONAI transforms, so
should work with almost all existing pipelines simply by setting `lazy=True`
on MONAI `Compose` instances.  See the
[Lazy Resampling topic](https://docs.monai.io/en/stable/lazy_resampling.html)
in the documentation for more details.

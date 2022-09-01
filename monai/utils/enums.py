# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from enum import Enum
from typing import Optional

from monai.utils.deprecate_utils import deprecated

__all__ = [
    "StrEnum",
    "NumpyPadMode",
    "GridSampleMode",
    "SplineMode",
    "InterpolateMode",
    "UpsampleMode",
    "BlendMode",
    "PytorchPadMode",
    "NdimageMode",
    "GridSamplePadMode",
    "Average",
    "MetricReduction",
    "LossReduction",
    "DiceCEReduction",
    "Weight",
    "ChannelMatching",
    "SkipMode",
    "Method",
    "TraceKeys",
    "InverseKeys",
    "CommonKeys",
    "GanKeys",
    "PostFix",
    "ForwardMode",
    "TransformBackends",
    "BoxModeName",
    "GridPatchSort",
    "FastMRIKeys",
    "SpaceKeys",
    "MetaKeys",
    "ColorOrder",
    "EngineStatsKeys",
]


class StrEnum(str, Enum):
    """
    Enum subclass that converts its value to a string.

    .. code-block:: python

        from monai.utils import StrEnum

        class Example(StrEnum):
            MODE_A = "A"
            MODE_B = "B"

        assert (list(Example) == ["A", "B"])
        assert Example.MODE_A == "A"
        assert str(Example.MODE_A) == "A"
        assert monai.utils.look_up_option("A", Example) == "A"
    """

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class NumpyPadMode(StrEnum):
    """
    See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    CONSTANT = "constant"
    EDGE = "edge"
    LINEAR_RAMP = "linear_ramp"
    MAXIMUM = "maximum"
    MEAN = "mean"
    MEDIAN = "median"
    MINIMUM = "minimum"
    REFLECT = "reflect"
    SYMMETRIC = "symmetric"
    WRAP = "wrap"
    EMPTY = "empty"


class NdimageMode(StrEnum):
    """
    The available options determine how the input array is extended beyond its boundaries when interpolating.
    See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
    """

    REFLECT = "reflect"
    GRID_MIRROR = "grid-mirror"
    CONSTANT = "constant"
    GRID_CONSTANT = "grid-constant"
    NEAREST = "nearest"
    MIRROR = "mirror"
    GRID_WRAP = "grid-wrap"
    WRAP = "wrap"


class GridSampleMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

    interpolation mode of `torch.nn.functional.grid_sample`

    Note:
        (documentation from `torch.nn.functional.grid_sample`)
        `mode='bicubic'` supports only 4-D input.
        When `mode='bilinear'` and the input is 5-D, the interpolation mode used internally will actually be trilinear.
        However, when the input is 4-D, the interpolation mode will legitimately be bilinear.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


class SplineMode(StrEnum):
    """
    Order of spline interpolation.

    See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
    """

    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


class InterpolateMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    AREA = "area"


class UpsampleMode(StrEnum):
    """
    See also: :py:class:`monai.networks.blocks.UpSample`
    """

    DECONV = "deconv"
    NONTRAINABLE = "nontrainable"  # e.g. using torch.nn.Upsample
    PIXELSHUFFLE = "pixelshuffle"


class BlendMode(StrEnum):
    """
    See also: :py:class:`monai.data.utils.compute_importance_map`
    """

    CONSTANT = "constant"
    GAUSSIAN = "gaussian"


class PytorchPadMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    """

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


class GridSamplePadMode(StrEnum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    """

    ZEROS = "zeros"
    BORDER = "border"
    REFLECTION = "reflection"


class Average(StrEnum):
    """
    See also: :py:class:`monai.metrics.rocauc.compute_roc_auc`
    """

    MACRO = "macro"
    WEIGHTED = "weighted"
    MICRO = "micro"
    NONE = "none"


class MetricReduction(StrEnum):
    """
    See also: :py:func:`monai.metrics.utils.do_metric_reduction`
    """

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"
    MEAN_BATCH = "mean_batch"
    SUM_BATCH = "sum_batch"
    MEAN_CHANNEL = "mean_channel"
    SUM_CHANNEL = "sum_channel"


class LossReduction(StrEnum):
    """
    See also:
        - :py:class:`monai.losses.dice.DiceLoss`
        - :py:class:`monai.losses.dice.GeneralizedDiceLoss`
        - :py:class:`monai.losses.focal_loss.FocalLoss`
        - :py:class:`monai.losses.tversky.TverskyLoss`
    """

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


class DiceCEReduction(StrEnum):
    """
    See also:
        - :py:class:`monai.losses.dice.DiceCELoss`
    """

    MEAN = "mean"
    SUM = "sum"


class Weight(StrEnum):
    """
    See also: :py:class:`monai.losses.dice.GeneralizedDiceLoss`
    """

    SQUARE = "square"
    SIMPLE = "simple"
    UNIFORM = "uniform"


class ChannelMatching(StrEnum):
    """
    See also: :py:class:`monai.networks.nets.HighResBlock`
    """

    PAD = "pad"
    PROJECT = "project"


class SkipMode(StrEnum):
    """
    See also: :py:class:`monai.networks.layers.SkipConnection`
    """

    CAT = "cat"
    ADD = "add"
    MUL = "mul"


class Method(StrEnum):
    """
    See also: :py:class:`monai.transforms.croppad.array.SpatialPad`
    """

    SYMMETRIC = "symmetric"
    END = "end"


class ForwardMode(StrEnum):
    """
    See also: :py:class:`monai.transforms.engines.evaluator.Evaluator`
    """

    TRAIN = "train"
    EVAL = "eval"


class TraceKeys(StrEnum):
    """Extra metadata keys used for traceable transforms."""

    CLASS_NAME: str = "class"
    ID: str = "id"
    ORIG_SIZE: str = "orig_size"
    EXTRA_INFO: str = "extra_info"
    DO_TRANSFORM: str = "do_transforms"
    KEY_SUFFIX: str = "_transforms"
    NONE: str = "none"


@deprecated(since="0.8.0", msg_suffix="use monai.utils.enums.TraceKeys instead.")
class InverseKeys:
    """
    Extra metadata keys used for inverse transforms.

    .. deprecated:: 0.8.0
        Use :class:`monai.utils.enums.TraceKeys` instead.

    """

    CLASS_NAME = "class"
    ID = "id"
    ORIG_SIZE = "orig_size"
    EXTRA_INFO = "extra_info"
    DO_TRANSFORM = "do_transforms"
    KEY_SUFFIX = "_transforms"
    NONE = "none"


class CommonKeys(StrEnum):
    """
    A set of common keys for dictionary based supervised training process.
    `IMAGE` is the input image data.
    `LABEL` is the training or evaluation label of segmentation or classification task.
    `PRED` is the prediction data of model output.
    `LOSS` is the loss value of current iteration.
    `INFO` is some useful information during training or evaluation, like loss value, etc.

    """

    IMAGE = "image"
    LABEL = "label"
    PRED = "pred"
    LOSS = "loss"
    METADATA = "metadata"


class GanKeys(StrEnum):
    """
    A set of common keys for generative adversarial networks.

    """

    REALS = "reals"
    FAKES = "fakes"
    LATENTS = "latents"
    GLOSS = "g_loss"
    DLOSS = "d_loss"


class PostFix(StrEnum):
    """Post-fixes."""

    @staticmethod
    def _get_str(prefix, suffix):
        return suffix if prefix is None else f"{prefix}_{suffix}"

    @staticmethod
    def meta(key: Optional[str] = None):
        return PostFix._get_str(key, "meta_dict")

    @staticmethod
    def orig_meta(key: Optional[str] = None):
        return PostFix._get_str(key, "orig_meta_dict")

    @staticmethod
    def transforms(key: Optional[str] = None):
        return PostFix._get_str(key, TraceKeys.KEY_SUFFIX[1:])


class TransformBackends(StrEnum):
    """
    Transform backends.
    """

    TORCH = "torch"
    NUMPY = "numpy"


class JITMetadataKeys(StrEnum):
    """
    Keys stored in the metadata file for saved Torchscript models. Some of these are generated by the routines
    and others are optionally provided by users.
    """

    NAME = "name"
    TIMESTAMP = "timestamp"
    VERSION = "version"
    DESCRIPTION = "description"


class BoxModeName(StrEnum):
    """
    Box mode names.
    """

    XYXY = "xyxy"  # [xmin, ymin, xmax, ymax]
    XYZXYZ = "xyzxyz"  # [xmin, ymin, zmin, xmax, ymax, zmax]
    XXYY = "xxyy"  # [xmin, xmax, ymin, ymax]
    XXYYZZ = "xxyyzz"  # [xmin, xmax, ymin, ymax, zmin, zmax]
    XYXYZZ = "xyxyzz"  # [xmin, ymin, xmax, ymax, zmin, zmax]
    XYWH = "xywh"  # [xmin, ymin, xsize, ysize]
    XYZWHD = "xyzwhd"  # [xmin, ymin, zmin, xsize, ysize, zsize]
    CCWH = "ccwh"  # [xcenter, ycenter, xsize, ysize]
    CCCWHD = "cccwhd"  # [xcenter, ycenter, zcenter, xsize, ysize, zsize]


class ProbMapKeys(StrEnum):
    """
    The keys to be used for generating the probability maps from patches
    """

    LOCATION = "mask_location"
    SIZE = "mask_size"
    COUNT = "num_patches"
    NAME = "name"


class GridPatchSort(StrEnum):
    """
    The sorting method for the generated patches in `GridPatch`
    """

    RANDOM = "random"
    MIN = "min"
    MAX = "max"

    @staticmethod
    def min_fn(x):
        return x[0].sum()

    @staticmethod
    def max_fn(x):
        return -x[0].sum()

    @staticmethod
    def get_sort_fn(sort_fn):
        if sort_fn == GridPatchSort.RANDOM:
            return random.random
        elif sort_fn == GridPatchSort.MIN:
            return GridPatchSort.min_fn
        elif sort_fn == GridPatchSort.MAX:
            return GridPatchSort.max_fn
        else:
            raise ValueError(
                f'sort_fn should be one of the following values, "{sort_fn}" was given:',
                [e.value for e in GridPatchSort],
            )


class WSIPatchKeys(StrEnum):
    """
    The keys to be used for metadata of patches extracted from whole slide images
    """

    LOCATION = "location"
    LEVEL = "level"
    SIZE = "size"
    COUNT = "count"
    PATH = "path"


class FastMRIKeys(StrEnum):
    """
    The keys to be used for extracting data from the fastMRI dataset
    """

    KSPACE = "kspace"
    MASK = "mask"
    FILENAME = "filename"
    RECON = "reconstruction_rss"
    ACQUISITION = "acquisition"
    MAX = "max"
    NORM = "norm"
    PID = "patient_id"


class SpaceKeys(StrEnum):
    """
    The coordinate system keys, for example, Nifti1 uses Right-Anterior-Superior or "RAS",
    DICOM (0020,0032) uses Left-Posterior-Superior or "LPS". This type does not distinguish spatial 1/2/3D.
    """

    RAS = "RAS"
    LPS = "LPS"


class MetaKeys(StrEnum):
    """
    Typical keys for MetaObj.meta
    """

    AFFINE = "affine"  # MetaTensor.affine
    ORIGINAL_AFFINE = "original_affine"  # the affine after image loading before any data processing
    SPATIAL_SHAPE = "spatial_shape"  # optional key for the length in each spatial dimension
    SPACE = "space"  # possible values of space type are defined in `SpaceKeys`
    ORIGINAL_CHANNEL_DIM = "original_channel_dim"  # an integer or "no_channel"


class ColorOrder(StrEnum):
    """
    Enums for color order. Expand as necessary.
    """

    RGB = "RGB"
    BGR = "BGR"


class EngineStatsKeys(StrEnum):
    """
    Default keys for the statistics of trainer and evaluator engines.

    """

    RANK = "rank"
    CURRENT_ITERATION = "current_iteration"
    CURRENT_EPOCH = "current_epoch"
    TOTAL_EPOCHS = "total_epochs"
    TOTAL_ITERATIONS = "total_iterations"
    BEST_VALIDATION_EPOCH = "best_validation_epoch"
    BEST_VALIDATION_METRIC = "best_validation_metric"


class DataStatsKeys(StrEnum):
    """
    Defaults keys for dataset statistical analysis modules

    """

    SUMMARY = "stats_summary"
    BY_CASE = "stats_by_cases"
    BY_CASE_IMAGE_PATH = "image_filepath"
    BY_CASE_LABEL_PATH = "label_filepath"
    IMAGE_STATS = "image_stats"
    FG_IMAGE_STATS = "image_foreground_stats"
    LABEL_STATS = "label_stats"


class ImageStatsKeys(StrEnum):
    """
    Defaults keys for dataset statistical analysis image modules

    """

    SHAPE = "shape"
    CHANNELS = "channels"
    CROPPED_SHAPE = "cropped_shape"
    SPACING = "spacing"
    INTENSITY = "intensity"


class LabelStatsKeys(StrEnum):
    """
    Defaults keys for dataset statistical analysis label modules

    """

    LABEL_UID = "labels"
    PIXEL_PCT = "pixel_percentage"
    IMAGE_INTST = "image_intensity"
    LABEL = "label"
    LABEL_SHAPE = "shape"
    LABEL_NCOMP = "ncomponents"

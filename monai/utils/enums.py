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

from enum import Enum
from typing import Optional

from monai.utils.deprecate_utils import deprecated

__all__ = [
    "NumpyPadMode",
    "GridSampleMode",
    "InterpolateMode",
    "UpsampleMode",
    "BlendMode",
    "PytorchPadMode",
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
    "PostFix",
    "ForwardMode",
    "TransformBackends",
    "BoxModeName",
    "GridPatchSort",
]


class NumpyPadMode(Enum):
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


class GridSampleMode(Enum):
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


class InterpolateMode(Enum):
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


class UpsampleMode(Enum):
    """
    See also: :py:class:`monai.networks.blocks.UpSample`
    """

    DECONV = "deconv"
    NONTRAINABLE = "nontrainable"  # e.g. using torch.nn.Upsample
    PIXELSHUFFLE = "pixelshuffle"


class BlendMode(Enum):
    """
    See also: :py:class:`monai.data.utils.compute_importance_map`
    """

    CONSTANT = "constant"
    GAUSSIAN = "gaussian"


class PytorchPadMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    """

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


class GridSamplePadMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    """

    ZEROS = "zeros"
    BORDER = "border"
    REFLECTION = "reflection"


class Average(Enum):
    """
    See also: :py:class:`monai.metrics.rocauc.compute_roc_auc`
    """

    MACRO = "macro"
    WEIGHTED = "weighted"
    MICRO = "micro"
    NONE = "none"


class MetricReduction(Enum):
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


class LossReduction(Enum):
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


class DiceCEReduction(Enum):
    """
    See also:
        - :py:class:`monai.losses.dice.DiceCELoss`
    """

    MEAN = "mean"
    SUM = "sum"


class Weight(Enum):
    """
    See also: :py:class:`monai.losses.dice.GeneralizedDiceLoss`
    """

    SQUARE = "square"
    SIMPLE = "simple"
    UNIFORM = "uniform"


class ChannelMatching(Enum):
    """
    See also: :py:class:`monai.networks.nets.HighResBlock`
    """

    PAD = "pad"
    PROJECT = "project"


class SkipMode(Enum):
    """
    See also: :py:class:`monai.networks.layers.SkipConnection`
    """

    CAT = "cat"
    ADD = "add"
    MUL = "mul"


class Method(Enum):
    """
    See also: :py:class:`monai.transforms.croppad.array.SpatialPad`
    """

    SYMMETRIC = "symmetric"
    END = "end"


class ForwardMode(Enum):
    """
    See also: :py:class:`monai.transforms.engines.evaluator.Evaluator`
    """

    TRAIN = "train"
    EVAL = "eval"


class TraceKeys:
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


class CommonKeys:
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


class PostFix:
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


class TransformBackends(Enum):
    """
    Transform backends.
    """

    TORCH = "torch"
    NUMPY = "numpy"


class JITMetadataKeys(Enum):
    """
    Keys stored in the metadata file for saved Torchscript models. Some of these are generated by the routines
    and others are optionally provided by users.
    """

    NAME = "name"
    TIMESTAMP = "timestamp"
    VERSION = "version"
    DESCRIPTION = "description"


class BoxModeName(Enum):
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


class GridPatchSort(Enum):
    """
    The sorting method for the generated patches in `GridPatch`
    """

    RANDOM = "random"
    MIN = "min"
    MAX = "max"

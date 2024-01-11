"""
A collection of "operator" transforms for spatial operations.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import torch

import monai
from monai.config import USE_COMPILED
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import AFFINE_TOL, compute_shape_offset, to_affine_nd
from monai.networks.layers import AffineTransform
from monai.transforms.croppad.array import ResizeWithPadOrCrop
from monai.transforms.intensity.array import GaussianSmooth
from monai.transforms.inverse import TraceableTransform
from monai.transforms.utils import create_rotate, create_translate, resolves_modes, scale_affine
from monai.transforms.utils_pytorch_numpy_unification import allclose
from monai.utils import (
    LazyAttr,
    TraceKeys,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    optional_import,
)

nib, has_nib = optional_import("nibabel")
cupy, _ = optional_import("cupy")
cupy_ndi, _ = optional_import("cupyx.scipy.ndimage")
np_ndi, _ = optional_import("scipy.ndimage")

__all__ = ["spatial_resample", "orientation", "flip", "resize", "rotate", "zoom", "rotate90", "affine_func"]


class Operator(ABC):
    """
    An abstract class defines APIs to load image files.

    Typical usage of an implementation of this class is:

    .. code-block:: python

        image_reader = MyImageReader()
        img_obj = image_reader.read(path_to_image)
        img_data, meta_data = image_reader.get_data(img_obj)

    - The `read` call converts image filenames into image objects,
    - The `get_data` call fetches the image data, as well as metadata.
    - A reader should implement `verify_suffix` with the logic of checking the input filename
      by the filename extensions.

    """
    @abstractmethod
    def apply(self, data: Sequence[PathLike] | PathLike, **kwargs) -> Sequence[Any] | Any:
        """
        Read image data from specified file or files.
        Note that it returns a data object or a sequence of data objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

class FlipBoxOp(Operator):
    def __init__(self) -> None:
        super().__init__()
    
    def apply(self, img, sp_axes, lazy, transform_info):
        """
        Functional implementation of flip.
        This function operates eagerly or lazily according to
        ``lazy`` (default ``False``).

        Args:
            img: data to be changed, assuming `img` is channel-first.
            sp_axes: spatial axes along which to flip over.
                If None, will flip over all of the axes of the input array.
                If axis is negative it counts from the last to the first axis.
                If axis is a tuple of ints, flipping is performed on all of the axes
                specified in the tuple.
            lazy: a flag that indicates whether the operation should be performed lazily or not
            transform_info: a dictionary with the relevant information pertaining to an applied transform.
        """
        sp_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
        sp_size = convert_to_numpy(sp_size, wrap_sequence=True).tolist()
        extra_info = {"axes": sp_axes}  # track the spatial axes
        axes = monai.transforms.utils.map_spatial_axes(img.ndim, sp_axes)  # use the axes with channel dim
        rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else torch.tensor(3.0, dtype=torch.double)
        # axes include the channel dim
        xform = torch.eye(int(rank) + 1, dtype=torch.double)
        for axis in axes:
            sp = axis - 1
            xform[sp, sp], xform[sp, -1] = xform[sp, sp] * -1, sp_size[sp] - 1
        meta_info = TraceableTransform.track_transform_meta(
            img, sp_size=sp_size, affine=xform, extra_info=extra_info, transform_info=transform_info, lazy=lazy
        )
        out = _maybe_new_metatensor(img)
        if lazy:
            return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
        out = torch.flip(out, axes)
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out


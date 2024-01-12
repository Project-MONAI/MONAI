"""
A collection of "operator" transforms for spatial operations.
"""

from __future__ import annotations

from collections.abc import Sequence
from abc import ABC, abstractmethod

import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import (
    optional_import,
)
from monai.transforms.spatial.functional import flip
from monai.apps.detection.transforms.box_ops import flip_boxes

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
    def apply(self, img: torch.Tensor, **kwargs):
        """
        Read image data from specified file or files.
        Note that it returns a data object or a sequence of data objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

class FlipImageOp(Operator):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, img, sp_axes, lazy, transform_info, **kwargs):
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
        return flip(img, sp_axes, lazy=lazy, transform_info=transform_info)  # type: ignore


class FlipPointOp(Operator):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, boxes: NdarrayOrTensor, spatial_size: Sequence[int] | int, spatial_axis: Sequence[int] | int | None = None, **kwargs):
        """
        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            spatial_size: image spatial size.
        """

        return flip_boxes(boxes, spatial_size=spatial_size, spatial_axis=spatial_axis)

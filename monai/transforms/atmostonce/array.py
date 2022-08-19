from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

import torch

from monai.config import DtypeLike, NdarrayOrTensor

from monai.transforms import InvertibleTransform, RandomizableTransform

from monai.transforms.atmostonce.apply import apply
from monai.transforms.atmostonce.functional import resize, rotate, zoom, spacing, croppad
from monai.transforms.atmostonce.lazy_transform import LazyTransform

from monai.utils import (GridSampleMode, GridSamplePadMode,
                         InterpolateMode, NumpyPadMode, PytorchPadMode)
from monai.utils.mapping_stack import MetaMatrix
from monai.utils.misc import ensure_tuple


# TODO: these transforms are intended to replace array transforms once development is done

# spatial
# =======

# TODO: why doesn't Spacing have antialiasing options?
class Spacing(LazyTransform, InvertibleTransform):

    def __init__(
        self,
        pixdim: Union[Sequence[float], float, np.ndarray],
        src_pixdim: Optional[Union[Sequence[float], float, np.ndarray]],
        diagonal: Optional[bool] = False,
        mode: Optional[Union[GridSampleMode, str]] = GridSampleMode.BILINEAR,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = GridSamplePadMode.BORDER,
        align_corners: Optional[bool] = False,
        dtype: Optional[DtypeLike] = np.float64,
        lazy_evaluation: Optional[bool] = False,
        shape_override: Optional[Sequence] = None
    ):
        LazyTransform.__init__(self, lazy_evaluation)
        self.pixdim = pixdim
        self.src_pixdim = src_pixdim
        self.diagonal = diagonal
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        align_corners: Optional[bool] = None,
        dtype: DtypeLike = None,
        shape_override: Optional[Sequence] = None
    ):

        mode_ = mode or self.mode
        padding_mode_ = padding_mode or self.padding_mode
        align_corners_ = align_corners or self.align_corners
        dtype_ = dtype or self.dtype

        img_t, transform, metadata = spacing(img, self.pixdim, self.src_pixdim, self.diagonal,
                                             mode_, padding_mode_, align_corners_, dtype_,
                                             shape_override)

        # TODO: candidate for refactoring into a LazyTransform method
        img_t.push_pending_transform(MetaMatrix(transform, metadata))
        if not self.lazy_evaluation:
            img_t = apply(img_t)

        return img_t

    def inverse(self, data):
        raise NotImplementedError()


class Resize(LazyTransform, InvertibleTransform):

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        size_mode: Optional[str] = "all",
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = False,
        anti_aliasing: Optional[bool] = False,
        anti_aliasing_sigma: Optional[Union[Sequence[float], float, None]] = None,
        dtype: Optional[Union[DtypeLike, torch.dtype]] = np.float32,
        lazy_evaluation: Optional[bool] = False
    ):
        LazyTransform.__init__(self, lazy_evaluation)
        self.spatial_size = spatial_size
        self.size_mode = size_mode
        self.mode = mode,
        self.align_corners = align_corners
        self.anti_aliasing = anti_aliasing
        self.anti_aliasing_sigma = anti_aliasing_sigma
        self.dtype = dtype

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[InterpolateMode, str]] = None,
        align_corners: Optional[bool] = None,
        anti_aliasing: Optional[bool] = None,
        anti_aliasing_sigma: Union[Sequence[float], float, None] = None,
        shape_override: Optional[Sequence] = None
    ) -> NdarrayOrTensor:
        mode_ = mode or self.mode
        align_corners_ = align_corners or self.align_corners
        anti_aliasing_ = anti_aliasing or self.anti_aliasing
        anti_aliasing_sigma_ = anti_aliasing_sigma or self.anti_aliasing_sigma

        img_t, transform, metadata = resize(img, self.spatial_size, self.size_mode, mode_,
                                            align_corners_, anti_aliasing_, anti_aliasing_sigma_,
                                            self.dtype, shape_override)

        # TODO: candidate for refactoring into a LazyTransform method
        img_t.push_pending_transform(MetaMatrix(transform, metadata))
        if not self.lazy_evaluation:
            img_t = apply(img_t)

        return img_t


class Rotate(LazyTransform, InvertibleTransform):

    def __init__(
        self,
        angle: Union[Sequence[float], float],
        keep_size: bool = True,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: Union[DtypeLike, torch.dtype] = np.float32,
        lazy_evaluation: Optional[bool] = False
    ):
        LazyTransform.__init__(self, lazy_evaluation)
        self.angle = angle
        self.keep_size = keep_size
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[InterpolateMode, str]] = None,
        padding_mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None,
        align_corners: Optional[bool] = None,
        shape_override: Optional[Sequence] = None
    ) -> NdarrayOrTensor:
        angle = self.angle
        mode = mode or self.mode
        padding_mode = padding_mode or self.padding_mode
        align_corners = align_corners or self.align_corners
        keep_size = self.keep_size
        dtype = self.dtype

        img_t, transform, metadata = rotate(img, angle, keep_size, mode, padding_mode,
                                            align_corners, dtype, shape_override)

        # TODO: candidate for refactoring into a LazyTransform method
        img_t.push_pending_transform(MetaMatrix(transform, metadata))
        if not self.lazy_evaluation:
            img_t = apply(img_t)

        return img_t

    def inverse(self, data):
        raise NotImplementedError()


class Zoom(LazyTransform, InvertibleTransform):
    """
    Zoom into / out of the image applying the `zoom` factor as a scalar, or if `zoom` is a tuple of
    values, apply each zoom factor to the appropriate dimension.
    """

    def __init__(
        self,
        zoom: Union[Sequence[float], float],
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        padding_mode: Union[NumpyPadMode, PytorchPadMode, str] = NumpyPadMode.EDGE,
        align_corners: Optional[bool] = None,
        keep_size: bool = True,
        dtype: Union[DtypeLike, torch.dtype] = np.float32,
        **kwargs
    ):
        self.zoom = zoom
        self.mode: InterpolateMode = InterpolateMode(mode)
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.keep_size = keep_size
        self.dtype = dtype
        self.kwargs = kwargs

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[InterpolateMode, str]] = None,
        padding_mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None,
        align_corners: Optional[bool] = None,
        shape_override: Optional[Sequence] = None
    ) -> NdarrayOrTensor:

        mode = self.mode or mode
        padding_mode = self.padding_mode or padding_mode
        align_corners = self.align_corners or align_corners
        keep_size = self.keep_size
        dtype = self.dtype

        img_t, transform, metadata = zoom(img, self.zoom, mode, padding_mode, align_corners,
                                          keep_size, dtype, shape_override)

        # TODO: candidate for refactoring into a LazyTransform method
        img_t.push_pending_transform(MetaMatrix(transform, metadata))
        if not self.lazy_evaluation:
            img_t = apply(img_t)

        return img_t

    def inverse(self, data):
        raise NotImplementedError()


class RandRotate(RandomizableTransform, InvertibleTransform, LazyTransform):

    def __init__(
        self,
        range_x: Optional[Union[Tuple[float, float], float]] = 0.0,
        range_y: Optional[Union[Tuple[float, float], float]] = 0.0,
        range_z: Optional[Union[Tuple[float, float], float]] = 0.0,
        prob: Optional[float] = 0.1,
        keep_size: bool = True,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: Union[DtypeLike, torch.dtype] = np.float32
    ):
        RandomizableTransform.__init__(self, prob)
        self.range_x = ensure_tuple(range_x)
        if len(self.range_x) == 1:
            self.range_x = tuple(sorted([-self.range_x[0], self.range_x[0]]))
        self.range_y = ensure_tuple(range_y)
        if len(self.range_y) == 1:
            self.range_y = tuple(sorted([-self.range_y[0], self.range_y[0]]))
        self.range_z = ensure_tuple(range_z)
        if len(self.range_z) == 1:
            self.range_z = tuple(sorted([-self.range_z[0], self.range_z[0]]))

        self.keep_size = keep_size
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.x = self.R.uniform(low=self.range_x[0], high=self.range_x[1])
        self.y = self.R.uniform(low=self.range_y[0], high=self.range_y[1])
        self.z = self.R.uniform(low=self.range_z[0], high=self.range_z[1])

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[InterpolateMode, str]] = None,
        padding_mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None,
        align_corners: Optional[bool] = None,
        dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
        randomize: Optional[bool] = True,
        get_matrix: Optional[bool] = False,
        shape_override: Optional[Sequence] = None

    ) -> NdarrayOrTensor:

        if randomize:
            self.randomize()

        img_dims = len(img.shape) - 1
        if self._do_transform:
            angle = self.x if img_dims == 2 else (self.x, self.y, self.z)
        else:
            angle = 0 if img_dims == 2 else (0, 0, 0)

        mode = self.mode or mode
        padding_mode = self.padding_mode or padding_mode
        align_corners = self.align_corners or align_corners
        keep_size = self.keep_size
        dtype = self.dtype

        img_t, transform, metadata = rotate(img, angle, keep_size, mode, padding_mode,
                                            align_corners, dtype, shape_override)

        # TODO: candidate for refactoring into a LazyTransform method
        img_t.push_pending_transform(MetaMatrix(transform, metadata))
        if not self.lazy_evaluation:
            img_t = apply(img_t)

        return img_t

    def inverse(
            self,
            data: NdarrayOrTensor,
    ):
        raise NotImplementedError()

# croppad
# =======


class CropPad(LazyTransform, InvertibleTransform):

    def __init__(
            self,
            slices: Sequence[slice],
            padmode: Optional[Union[GridSamplePadMode, str]] = GridSamplePadMode.BORDER,
            lazy_evaluation: Optional[bool] = True,
    ):
        LazyTransform.__init__(self, lazy_evaluation)
        self.slices = slices
        self.padmode = padmode

    def __call__(
            self,
            img: NdarrayOrTensor,
            shape_override: Optional[Sequence] = None
    ):

        img_t, transform, metadata = croppad(img, self.slices, self.padmode, shape_override)

        # TODO: candidate for refactoring into a LazyTransform method
        img_t.push_pending_transform(MetaMatrix(transform, metadata))
        if not self.lazy_evaluation:
            img_t = apply(img_t)

        return img_t

    def inverse(
            self,
            data: NdarrayOrTensor
    ):
        raise NotImplementedError()

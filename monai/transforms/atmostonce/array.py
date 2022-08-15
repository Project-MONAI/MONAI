from typing import Optional, Sequence, Tuple, Union

import numpy as np

import torch

from monai.config import DtypeLike, NdarrayOrTensor

from monai.transforms import Transform

from monai.utils import (GridSampleMode, GridSamplePadMode,
                         InterpolateMode, NumpyPadMode, PytorchPadMode)
from monai.utils.mapping_stack import MappingStack, MatrixFactory
from monai.utils.misc import get_backend_from_data, get_device_from_data


class Rotate(Transform):

    def __init__(
        self,
        angle: Union[Sequence[float], float],
        keep_size: bool = True,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: Union[DtypeLike, torch.dtype] = np.float32
    ):
        self.angle = angle
        self.keep_size = keep_size
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype

    def __call__(
        self,
        img: NdarrayOrTensor,
        mapping_stack: Optional[MappingStack] = None,
        mode: Optional[Union[InterpolateMode, str]] = None,
        padding_mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None,
        align_corners: Optional[bool] = None,
    ) -> NdarrayOrTensor:
        mode = self.mode if mode is None else mode
        padding_mode = self.padding_mode if padding_mode is None else padding_mode
        align_corners = self.align_corners if align_corners is None else align_corners
        keep_size = self.keep_size
        dtype = self.dtype
        matrix_factory = MatrixFactory(len(img.shape)-1,
                                       get_backend_from_data(img),
                                       get_device_from_data(img))
        if mapping_stack is None:
            mapping_stack = MappingStack(matrix_factory)
        mapping_stack.push(matrix_factory.rotate_euler(self.angle,
                                                       **{
                                                           "padding_mode": padding_mode,
                                                           "mode": mode,
                                                           "align_corners": align_corners,
                                                           "keep_size": keep_size,
                                                           "dtype": dtype
                                                       }))


class Zoom(Transform):
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
        mapping_stack: Optional[MappingStack] = None,
        mode: Optional[Union[InterpolateMode, str]] = None,
        padding_mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None,
        align_corners: Optional[bool] = None
    ) -> NdarrayOrTensor:

        mode = self.mode if mode is None else mode
        padding_mode = self.padding_mode if padding_mode is None else padding_mode
        align_corners = self.align_corners if align_corners is None else align_corners
        keep_size = self.keep_size
        dtype = self.dtype
        matrix_factory = MatrixFactory(len(img.shape)-1,
                                       get_backend_from_data(img),
                                       get_device_from_data(img))
        if mapping_stack is None:
            mapping_stack = MappingStack(matrix_factory)
        mapping_stack.push(matrix_factory.scale(self.zoom,
                                                **{
                                                    "padding_mode": padding_mode,
                                                    "mode": mode,
                                                    "align_corners": align_corners,
                                                    "keep_size": keep_size,
                                                    "dtype": dtype
                                                }))
        img.add


class Resize(Transform):

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        size_mode: str = "all",
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        anti_aliasing: bool = False,
        anti_aliasing_sigma: Union[Sequence[float], float, None] = None
    ):
        self.spatial_size = spatial_size
        self.size_mode = size_mode
        self.mode = mode,
        self.align_corners = align_corners
        self.anti_aliasing = anti_aliasing
        self.anti_aliasing_sigma = anti_aliasing_sigma

    def __call__(
        self,
        img: NdarrayOrTensor,
        mapping_stack: Optional[MappingStack] = None,
        mode: Optional[Union[InterpolateMode, str]] = None,
        align_corners: Optional[bool] = None,
        anti_aliasing: Optional[bool] = None,
        anti_aliasing_sigma: Union[Sequence[float], float, None] = None
    ) -> NdarrayOrTensor:
        mode = self.mode if mode is None else mode
        align_corners = self.align_corners if align_corners is None else align_corners
        keep_size = self.keep_size
        dtype = self.dtype
        matrix_factory = MatrixFactory(len(img.shape)-1,
                                       get_backend_from_data(img),
                                       get_device_from_data(img))
        if mapping_stack is None:
            mapping_stack = MappingStack(matrix_factory)
        mapping_stack.push(matrix_factory.scale(self.zoom,
                                                **{
                                                    "mode": mode,
                                                    "align_corners": align_corners,
                                                    "keep_size": keep_size,
                                                    "dtype": dtype
                                                }))


class Spacing(Transform):

    def __init__(
        self,
        pixdim: Union[Sequence[float], float, np.ndarray],
        diagonal: bool = False,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
        image_only: bool = False
    ):
        self.pixdim = pixdim
        self.diagonal = diagonal
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype
        self.image_only = image_only

    def __call__(
        self,
        img: NdarrayOrTensor,
        mapping_stack: Optional[MappingStack] = None,
        affine: Optional[NdarrayOrTensor] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        align_corners: Optional[bool] = None,
        dtype: DtypeLike = None,
        output_spatial_shape: Optional[Union[Sequence[int], np.ndarray, int]] = None
    ) -> Union[NdarrayOrTensor, Tuple[NdarrayOrTensor, NdarrayOrTensor, NdarrayOrTensor]]:
        pass



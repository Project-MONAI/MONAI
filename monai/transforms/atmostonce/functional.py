from typing import Optional, Sequence, Union

import numpy as np

import torch
from monai.transforms import create_rotate, create_translate, GaussianSmooth

from monai.data import get_track_meta
from monai.transforms.atmostonce.apply import extents_from_shape, shape_from_extents
from monai.utils import convert_to_tensor, get_equivalent_dtype, ensure_tuple_rep, convert_to_dst_type, look_up_option, \
    GridSampleMode, GridSamplePadMode, fall_back_tuple, ensure_tuple_size, ensure_tuple, InterpolateMode, NumpyPadMode

from monai.config import DtypeLike
from monai.utils.mapping_stack import MetaMatrix, MatrixFactory


def spacing(
    img: torch.Tensor,
    pixdim: Union[Sequence[float], float],
    src_pixdim: Union[Sequence[float], float],
    diagonal: Optional[bool] = False,
    mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.AREA,
    padding_mode: Optional[Union[NumpyPadMode, GridSamplePadMode, str]] = NumpyPadMode.EDGE,
    align_corners: Optional[bool] = False,
    dtype: Optional[Union[DtypeLike, torch.dtype]] = None
):
    """
    Args:
        img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
            ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``self.mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        anti_aliasing: bool, optional
            Whether to apply a Gaussian filter to smooth the image prior
            to downsampling. It is crucial to filter when downsampling
            the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
        anti_aliasing_sigma: {float, tuple of floats}, optional
            Standard deviation for Gaussian filtering used when anti-aliasing.
            By default, this value is chosen as (s - 1) / 2 where s is the
            downsampling factor, where s > 1. For the up-size case, s < 1, no
            anti-aliasing is performed prior to rescaling.

    Raises:
        ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

    """

    img_ = convert_to_tensor(img, track_meta=get_track_meta())
    input_ndim = len(img.shape) - 1

    pixdim_ = ensure_tuple_rep(pixdim, input_ndim)
    src_pixdim_ = ensure_tuple_rep(src_pixdim, input_ndim)

    if diagonal is True:
        raise ValueError("'diagonal' value of True is not currently supported")

    mode_ = look_up_option(mode, GridSampleMode)
    padding_mode_ = look_up_option(padding_mode, GridSamplePadMode)
    dtype_ = get_equivalent_dtype(dtype or img.dtype, torch.Tensor)
    zoom_factors = [i / j for i, j in zip(src_pixdim_, pixdim_)]

    transform = MatrixFactory.from_tensor(img).scale(zoom_factors)
    im_extents = extents_from_shape(img.shape)
    im_extents = [transform.matrix.matrix @ e for e in im_extents]
    spatial_shape_ = shape_from_extents(im_extents)

    metadata = {
        "pixdim": pixdim_,
        "src_pixdim": src_pixdim_,
        "diagonal": diagonal,
        "mode": mode_,
        "padding_mode": padding_mode_,
        "align_corners": align_corners,
        "dtype": dtype_,
        "im_extents": im_extents,
        "spatial_shape": spatial_shape_
    }
    return img_, transform, metadata


def orientation(
        img: torch.Tensor
):
    pass


def flip(
        img: torch.Tensor
):
    pass


def resize(
    img: torch.Tensor,
    spatial_size: Union[Sequence[int], int],
    size_mode: str = "all",
    mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.AREA,
    align_corners: Optional[bool] = False,
    anti_aliasing: Optional[bool] = None,
    anti_aliasing_sigma: Optional[Union[Sequence[float], float]] = None,
    dtype: Optional[Union[DtypeLike, torch.dtype]] = None
):
    """
    Args:
        img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
            ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``self.mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        anti_aliasing: bool, optional
            Whether to apply a Gaussian filter to smooth the image prior
            to downsampling. It is crucial to filter when downsampling
            the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
        anti_aliasing_sigma: {float, tuple of floats}, optional
            Standard deviation for Gaussian filtering used when anti-aliasing.
            By default, this value is chosen as (s - 1) / 2 where s is the
            downsampling factor, where s > 1. For the up-size case, s < 1, no
            anti-aliasing is performed prior to rescaling.

    Raises:
        ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

    """

    img_ = convert_to_tensor(img, track_meta=get_track_meta())
    input_ndim = len(img.shape) - 1

    if size_mode == "all":
        output_ndim = len(ensure_tuple(spatial_size))
        if output_ndim > input_ndim:
            input_shape = ensure_tuple_size(img.shape, output_ndim + 1, 1)
            img = img.reshape(input_shape)
        elif output_ndim < input_ndim:
            raise ValueError(
                "len(spatial_size) must be greater or equal to img spatial dimensions, "
                f"got spatial_size={output_ndim} img={input_ndim}."
            )
        spatial_size_ = fall_back_tuple(spatial_size, img.shape[1:])
    else:  # for the "longest" mode
        img_size = img.shape[1:]
        if not isinstance(spatial_size, int):
            raise ValueError("spatial_size must be an int number if size_mode is 'longest'.")
        scale = spatial_size / max(img_size)
        spatial_size_ = tuple(int(round(s * scale)) for s in img_size)

    mode_ = look_up_option(mode, GridSampleMode)
    dtype_ = get_equivalent_dtype(dtype or img.dtype, torch.Tensor)
    zoom_factors = [i / j for i, j in zip(spatial_size, img.shape[1:])]
    transform = MatrixFactory.from_tensor(img).scale(zoom_factors)
    im_extents = extents_from_shape(img.shape)
    im_extents = [transform.matrix.matrix @ e for e in im_extents]
    spatial_shape_ = shape_from_extents(im_extents)

    metadata = {
        "spatial_size": spatial_size_,
        "size_mode": size_mode,
        "mode": mode_,
        "align_corners": align_corners,
        "anti_aliasing": anti_aliasing,
        "anti_aliasing_sigma": anti_aliasing_sigma,
        "dtype": dtype_,
        "im_extents": im_extents,
        "spatial_shape": spatial_shape_
    }
    return img_, transform, metadata


def rotate(
    img: torch.Tensor,
    angle: Union[Sequence[float], float],
    keep_size: Optional[bool] = True,
    mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.AREA,
    padding_mode: Optional[Union[NumpyPadMode, GridSamplePadMode, str]] = NumpyPadMode.EDGE,
    align_corners: Optional[bool] = False,
    dtype: Optional[Union[DtypeLike, torch.dtype]] = None
):
    """
    Args:
        img: channel first array, must have shape: [chns, H, W] or [chns, H, W, D].
        angle: Rotation angle(s) in radians. should a float for 2D, three floats for 3D.
        keep_size: If it is True, the output shape is kept the same as the input.
            If it is False, the output shape is adapted so that the
            input array is contained completely in the output. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``self.mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``self.padding_mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Defaults to ``self.align_corners``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        align_corners: Defaults to ``self.align_corners``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        dtype: data type for resampling computation. Defaults to ``self.dtype``.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.

    Raises:
        ValueError: When ``img`` spatially is not one of [2D, 3D].

    """

    img_ = convert_to_tensor(img, track_meta=get_track_meta())
    mode_ = look_up_option(mode, GridSampleMode)
    padding_mode_ = look_up_option(padding_mode, GridSamplePadMode)
    dtype_ = get_equivalent_dtype(dtype or img.dtype, torch.Tensor)
    input_ndim = len(img_.shape) - 1
    if input_ndim not in (2, 3):
        raise ValueError(f"Unsupported image dimension: {input_ndim}, available options are [2, 3].")
    angle_ = ensure_tuple_rep(angle, 1 if input_ndim == 2 else 3)
    transform = create_rotate(input_ndim, angle_)
    im_extents = extents_from_shape(img.shape)
    if not keep_size:
        im_extents = [transform @ e for e in im_extents]
        spatial_shape = shape_from_extents(im_extents)
    else:
        spatial_shape = img_.shape

    metadata = {
        "angle": angle_,
        "keep_size": keep_size,
        "mode": mode_,
        "padding_mode": padding_mode_,
        "align_corners": align_corners,
        "dtype": dtype_,
        "im_extents": im_extents,
        "spatial_shape": spatial_shape
    }
    return img_, transform, metadata


def zoom(
        img: torch.Tensor,
        zoom: Union[Sequence[float], float],
        mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.AREA,
        padding_mode: Optional[Union[NumpyPadMode, GridSamplePadMode, str]] = NumpyPadMode.EDGE,
        align_corners: Optional[bool] = False,
        keep_size: Optional[bool] = True,
        dtype: Optional[Union[DtypeLike, torch.dtype]] = None
):
    """
    Args:
        img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
            ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``self.mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

    Raises:
        ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

    """

    img_ = convert_to_tensor(img, track_meta=get_track_meta())
    input_ndim = len(img.shape) - 1

    zoom_factors = ensure_tuple_rep(zoom, input_ndim)


    mode_ = look_up_option(mode, GridSampleMode)
    padding_mode_ = look_up_option(padding_mode, GridSamplePadMode)
    dtype_ = get_equivalent_dtype(dtype or img.dtype, torch.Tensor)

    transform = MatrixFactory.from_tensor(img).scale(zoom_factors)
    im_extents = extents_from_shape(img.shape)
    if keep_size is False:
        im_extents = [transform.matrix.matrix @ e for e in im_extents]
        spatial_shape_ = shape_from_extents(im_extents)
    else:
        spatial_shape_ = img_.shape

    metadata = {
        "zoom": zoom_factors,
        "mode": mode_,
        "padding_mode": padding_mode_,
        "align_corners": align_corners,
        "keep_size": keep_size,
        "dtype": dtype_
    }
    return img_, transform, metadata


def rotate90(
        img: torch.Tensor
):
    pass

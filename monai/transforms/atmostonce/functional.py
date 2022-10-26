from typing import Optional, Sequence, Tuple, Union

import numpy as np

import torch
from monai.networks.layers import GaussianFilter

from monai.networks.utils import meshgrid_ij

from monai.transforms import create_rotate, create_translate, map_spatial_axes, create_grid

from monai.data import get_track_meta
from monai.transforms.atmostonce.apply import extents_from_shape, shape_from_extents
from monai.utils import convert_to_tensor, get_equivalent_dtype, ensure_tuple_rep, look_up_option, \
    GridSampleMode, GridSamplePadMode, fall_back_tuple, ensure_tuple_size, ensure_tuple, InterpolateMode, NumpyPadMode

from monai.config import DtypeLike
from monai.utils.mapping_stack import MatrixFactory


def identity(
    img: torch.Tensor,
    mode: Optional[Union[InterpolateMode, str]] = None,
    padding_mode: Optional[Union[NumpyPadMode, GridSamplePadMode, str]] = None,
    dtype: Optional[Union[DtypeLike, torch.dtype]] = None
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    mode_ = None if mode is None else look_up_option(mode, GridSampleMode)
    padding_mode_ = None if padding_mode is None else look_up_option(padding_mode, GridSamplePadMode)
    dtype_ = get_equivalent_dtype(dtype or img_.dtype, torch.Tensor)

    transform = MatrixFactory.from_tensor(img_).identity().matrix.matrix

    metadata = dict()
    if mode_ is not None:
        metadata["mode"] = mode_
    if padding_mode_ is not None:
        metadata["padding_mode"] = padding_mode_
    metadata["dtype"] = dtype_
    return img_, transform, metadata


def spacing(
    img: torch.Tensor,
    pixdim: Union[Sequence[float], float],
    src_pixdim: Union[Sequence[float], float],
    diagonal: Optional[bool] = False,
    mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.AREA,
    padding_mode: Optional[Union[NumpyPadMode, GridSamplePadMode, str]] = NumpyPadMode.EDGE,
    align_corners: Optional[bool] = False,
    dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
    shape_override: Optional[Sequence] = None
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
    input_shape = img_.shape if shape_override is None else shape_override
    input_ndim = len(input_shape) - 1

    pixdim_ = ensure_tuple_rep(pixdim, input_ndim)
    src_pixdim_ = ensure_tuple_rep(src_pixdim, input_ndim)

    if diagonal is True:
        raise ValueError("'diagonal' value of True is not currently supported")

    mode_ = look_up_option(mode, GridSampleMode)
    padding_mode_ = look_up_option(padding_mode, GridSamplePadMode)
    dtype_ = get_equivalent_dtype(dtype or img.dtype, torch.Tensor)
    zoom_factors = [i / j for i, j in zip(src_pixdim_, pixdim_)]

    # TODO: decide whether we are consistently returning MetaMatrix or concrete transforms
    transform = MatrixFactory.from_tensor(img).scale(zoom_factors).matrix.matrix
    im_extents = extents_from_shape(input_shape)
    im_extents = [transform @ e for e in im_extents]
    shape_override_ = shape_from_extents(input_shape, im_extents)

    metadata = {
        "pixdim": pixdim_,
        "src_pixdim": src_pixdim_,
        "diagonal": diagonal,
        "mode": mode_,
        "padding_mode": padding_mode_,
        "align_corners": align_corners,
        "dtype": dtype_,
        "im_extents": im_extents,
        "shape_override": shape_override_
    }
    return img_, transform, metadata


def orientation(
        img: torch.Tensor
):
    pass


def flip(
        img: torch.Tensor,
        spatial_axis: Union[Sequence[int], int],
        shape_override: Optional[Sequence] = None
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())
    input_shape = img_.shape if shape_override is None else shape_override

    spatial_axis_ = spatial_axis
    if spatial_axis_ is None:
        spatial_axis_ = tuple(i for i in range(len(input_shape[1:])))
    transform = MatrixFactory.from_tensor(img).flip(spatial_axis_).matrix.matrix
    im_extents = extents_from_shape(input_shape)
    im_extents = [transform @ e for e in im_extents]

    shape_override_ = shape_from_extents(input_shape, im_extents)

    metadata = {
        "spatial_axes": spatial_axis,
        "im_extents": im_extents,
        "shape_override": shape_override_
    }
    return img_, transform, metadata


def resize(
    img: torch.Tensor,
    spatial_size: Union[Sequence[int], int],
    size_mode: str = "all",
    mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.AREA,
    align_corners: Optional[bool] = False,
    anti_aliasing: Optional[bool] = None,
    anti_aliasing_sigma: Optional[Union[Sequence[float], float]] = None,
    dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
    shape_override: Optional[Sequence] = None
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
    input_shape = img_.shape if shape_override is None else shape_override
    input_ndim = len(input_shape) - 1

    if size_mode == "all":
        output_ndim = len(ensure_tuple(spatial_size))
        if output_ndim > input_ndim:
            input_shape = ensure_tuple_size(input_shape, output_ndim + 1, 1)
            img = img.reshape(input_shape)
        elif output_ndim < input_ndim:
            raise ValueError(
                "len(spatial_size) must be greater or equal to img spatial dimensions, "
                f"got spatial_size={output_ndim} img={input_ndim}."
            )
        spatial_size_ = fall_back_tuple(spatial_size, input_shape[1:])
    else:  # for the "longest" mode
        img_size = input_shape[1:]
        if not isinstance(spatial_size, int):
            raise ValueError("spatial_size must be an int number if size_mode is 'longest'.")
        scale = spatial_size / max(img_size)
        spatial_size_ = tuple(int(round(s * scale)) for s in img_size)

    mode_ = look_up_option(mode, GridSampleMode)
    dtype_ = get_equivalent_dtype(dtype or img.dtype, torch.Tensor)
    zoom_factors = [i / j for i, j in zip(spatial_size, input_shape[1:])]
    transform = MatrixFactory.from_tensor(img).scale(zoom_factors).matrix.matrix
    im_extents = extents_from_shape(input_shape)
    im_extents = [transform @ e for e in im_extents]
    shape_override_ = shape_from_extents(input_shape, im_extents)

    metadata = {
        "spatial_size": spatial_size_,
        "size_mode": size_mode,
        "mode": mode_,
        "align_corners": align_corners,
        "anti_aliasing": anti_aliasing,
        "anti_aliasing_sigma": anti_aliasing_sigma,
        "dtype": dtype_,
        "im_extents": im_extents,
        "shape_override": shape_override_
    }
    return img_, transform, metadata


def rotate(
    img: torch.Tensor,
    angle: Union[Sequence[float], float],
    keep_size: Optional[bool] = True,
    mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.AREA,
    padding_mode: Optional[Union[NumpyPadMode, GridSamplePadMode, str]] = NumpyPadMode.EDGE,
    align_corners: Optional[bool] = False,
    dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
    shape_override: Optional[Sequence] = None
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
    input_shape = img_.shape if shape_override is None else shape_override
    input_ndim = len(input_shape) - 1
    if input_ndim not in (2, 3):
        raise ValueError(f"Unsupported image dimension: {input_ndim}, available options are [2, 3].")

    angle_ = ensure_tuple_rep(angle, 1 if input_ndim == 2 else 3)
    rotate_tx = torch.from_numpy(create_rotate(input_ndim, angle_))
    im_extents = extents_from_shape(input_shape)
    if not keep_size:
        im_extents = [rotate_tx @ e for e in im_extents]
        spatial_shape = shape_from_extents(input_shape, im_extents)
    else:
        spatial_shape = input_shape
    transform = rotate_tx
    metadata = {
        "angle": angle_,
        "keep_size": keep_size,
        "mode": mode_,
        "padding_mode": padding_mode_,
        "align_corners": align_corners,
        "dtype": dtype_,
        "im_extents": im_extents,
        "shape_override": spatial_shape
    }
    return img_, transform, metadata


def zoom(
        img: torch.Tensor,
        factor: Union[Sequence[float], float],
        mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.BILINEAR,
        padding_mode: Optional[Union[NumpyPadMode, GridSamplePadMode, str]] = NumpyPadMode.EDGE,
        align_corners: Optional[bool] = False,
        keep_size: Optional[bool] = True,
        dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
        shape_override: Optional[Sequence] = None
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
    input_shape = img_.shape if shape_override is None else shape_override
    input_ndim = len(input_shape) - 1

    zoom_factors = ensure_tuple_rep(factor, input_ndim)
    zoom_factors = [1 / f for f in zoom_factors]

    mode_ = look_up_option(mode, GridSampleMode)
    padding_mode_ = look_up_option(padding_mode, GridSamplePadMode)
    dtype_ = get_equivalent_dtype(dtype or img_.dtype, torch.Tensor)

    transform = MatrixFactory.from_tensor(img_).scale(zoom_factors).matrix.matrix
    im_extents = extents_from_shape(input_shape)
    if keep_size is False:
        im_extents = [transform @ e for e in im_extents]
        shape_override_ = shape_from_extents(input_shape, im_extents)
    else:
        shape_override_ = input_shape

    metadata = {
        "factor": zoom_factors,
        "mode": mode_,
        "padding_mode": padding_mode_,
        "align_corners": align_corners,
        "keep_size": keep_size,
        "dtype": dtype_,
        "im_extents": im_extents,
        "shape_override": shape_override_
    }
    return img_, transform, metadata


def rotate90(
        img: torch.Tensor,
        k: Optional[int] = 1,
        spatial_axes: Optional[Tuple[int, int]] = (0, 1),
        shape_override: Optional[bool] = None
):
    if len(spatial_axes) != 2:
        raise ValueError("'spatial_axes' must be a tuple of two integers indicating")

    img_ = convert_to_tensor(img, track_meta=get_track_meta())
    # axes = map_spatial_axes(img.ndim, spatial_axes)
    # ori_shape = img.shape[1:]
    input_shape = img_.shape if shape_override is None else shape_override
    input_ndim = len(input_shape) - 1

    transform = MatrixFactory.from_tensor(img_).rotate_90(k, )

    metadata = {
        "k": k,
        "spatial_axes": spatial_axes,
        "shape_override": shape_override
    }
    return img_, transform, metadata


def grid_distortion(
        img: torch.Tensor,
        num_cells: Union[Tuple[int], int],
        distort_steps: Sequence[Sequence[float]],
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        shape_override: Optional[Tuple[int]] = None
):
    all_ranges = []
    num_cells = ensure_tuple_rep(num_cells, len(img.shape) - 1)
    for dim_idx, dim_size in enumerate(img.shape[1:]):
        dim_distort_steps = distort_steps[dim_idx]
        ranges = torch.zeros(dim_size, dtype=torch.float32)
        cell_size = dim_size // num_cells[dim_idx]
        prev = 0
        for idx in range(num_cells[dim_idx] + 1):
            start = int(idx * cell_size)
            end = start + cell_size
            if end > dim_size:
                end = dim_size
                cur = dim_size
            else:
                cur = prev + cell_size * dim_distort_steps[idx]
            prev = cur
        ranges = range - (dim_size - 1.0) / 2.0
        all_ranges.append()
    coords = meshgrid_ij(*all_ranges)
    grid = torch.stack([*coords, torch.ones_like(coords[0])])

    metadata = {
        "num_cells": num_cells,
        "distort_steps": distort_steps,
        "mode": mode,
        "padding_mode": padding_mode
    }

    return img, grid, metadata


def elastic_3d(
        img: torch.Tensor,
        sigma: float,
        magnitude: float,
        offsets: torch.Tensor,
        spatial_size: Optional[Union[Tuple[int, int, int], int]] = None,
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        device: Optional[torch.device] = None,
        shape_override: Optional[Tuple[float]] = None
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    sp_size = fall_back_tuple(spatial_size, img.shape[1:])
    device_ = img.device if isinstance(img, torch.Tensor) else device
    grid = create_grid(spatial_size=sp_size, device=device_, backend="torch")
    gaussian = GaussianFilter(3, sigma, 3.0).to(device=device_)
    grid[:3] += gaussian(offsets)[0] * magnitude

    metadata = {
        "sigma": sigma,
        "magnitude": magnitude,
        "offsets": offsets,
    }
    if spatial_size is not None:
        metadata["spatial_size"] = spatial_size
    if mode is not None:
        metadata["mode"] = mode
    if padding_mode is not None:
        metadata["padding_mode"] = padding_mode
    if shape_override is not None:
        metadata["shape_override"] = shape_override

    return img_, grid, metadata


def translate(
        img: torch.Tensor,
        translation: Sequence[float],
        mode: Optional[Union[GridSampleMode, str]] = GridSampleMode.BILINEAR,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = NumpyPadMode.EDGE,
        dtype: Union[DtypeLike, torch.dtype] = np.float32,
        shape_override: Optional[Sequence] = None
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())
    input_shape = img_.shape if shape_override is None else shape_override
    input_ndim = len(input_shape) - 1
    if len(translation) != input_ndim:
        raise ValueError(f"'translate' length {len(translation)} must be equal to 'img' "
                         f"spatial dimensions of {input_ndim}")

    transform = MatrixFactory.from_tensor(img).translate(translation).matrix.matrix
    im_extents = extents_from_shape(input_shape)
    im_extents = [transform @ e for e in im_extents]
    # shape_override_ = shape_from_extents(input_shape, im_extents)

    metadata = {
        "translation": translation,
        "padding_mode": padding_mode,
        "dtype": img.dtype,
        "im_extents": im_extents,
        # "shape_override": shape_override_
    }
    return img_, transform, metadata


def croppad(
        img: torch.Tensor,
        slices: Union[Sequence[slice], slice],
        padding_mode: Optional[Union[GridSamplePadMode, str]] = NumpyPadMode.EDGE,
        shape_override: Optional[Sequence] = None
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())
    input_shape = img_.shape if shape_override is None else shape_override
    input_ndim = len(input_shape) - 1
    if len(slices) != input_ndim:
        raise ValueError(f"'slices' length {len(slices)} must be equal to 'img' "
                         f"spatial dimensions of {input_ndim}")

    img_centers = [i / 2 for i in input_shape[1:]]
    slice_centers = [(s.stop + s.start) / 2 for s in slices]
    deltas = [s - i for i, s in zip(img_centers, slice_centers)]
    transform = MatrixFactory.from_tensor(img).translate(deltas).matrix.matrix
    im_extents = extents_from_shape([input_shape[0]] + [s.stop - s.start for s in slices])
    im_extents = [transform @ e for e in im_extents]
    shape_override_ = shape_from_extents(input_shape, im_extents)

    metadata = {
        "slices": slices,
        "padding_mode": padding_mode,
        "dtype": img.dtype,
        "im_extents": im_extents,
        "shape_override": shape_override_
    }
    return img_, transform, metadata

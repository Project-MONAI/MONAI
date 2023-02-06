from __future__ import annotations

from typing import Sequence

import numpy as np

import torch
from monai.transforms.inverse import InvertibleTransform

from monai.transforms.utils import create_translate, create_grid, Transform

from monai.utils import deprecated_arg

from monai.networks.layers import grid_pull

from monai.transforms.utils_pytorch_numpy_unification import linalg_inv, moveaxis, where

from monai.data.utils import to_affine_nd
from monai.data.meta_tensor import MetaTensor, get_track_meta

from monai.config import NdarrayOrTensor, DtypeLike, USE_COMPILED
from monai.transforms.lazy.utils import AffineMatrix
from monai.utils import LazyAttr, convert_data_type, convert_to_dst_type, TraceKeys, convert_to_tensor, fall_back_tuple, \
    GridSampleMode, GridSamplePadMode, TransformBackends, look_up_option, convert_to_cupy, convert_to_numpy, \
    optional_import, SplineMode, NdimageMode

cupy, _ = optional_import("cupy")
cupy_ndi, _ = optional_import("cupyx.scipy.ndimage")
np_ndi, _ = optional_import("scipy.ndimage")


def resample(data: torch.Tensor, matrix: NdarrayOrTensor, kwargs: dict | None = None):
    """
    This is a minimal implementation of resample that always uses Affine.
    """
    if not AffineMatrix.is_affine_shaped(matrix):
        raise NotImplementedError("calling dense grid resample API not implemented")
    kwargs = {} if kwargs is None else kwargs
    init_kwargs = {
        "spatial_size": kwargs.pop(LazyAttr.SHAPE, data.shape)[1:],
        "dtype": kwargs.pop(LazyAttr.DTYPE, data.dtype),
    }
    call_kwargs = {
        "mode": kwargs.pop(LazyAttr.INTERP_MODE, None),
        "padding_mode": kwargs.pop(LazyAttr.PADDING_MODE, None),
    }
    resampler = Resampler(affine=matrix, image_only=True, **init_kwargs)
    with resampler.trace_transform(False):  # don't track this transform in `data`
        return resampler(img=data, **call_kwargs)


class ResampleImpl:
    """
    TODO: incorporate what is needed into the new lazy resampling global resample method
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.BORDER,
        norm_coords: bool = True,
        device: torch.device | None = None,
        dtype: DtypeLike = np.float64,
    ) -> None:
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull (experimental).
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses an integer to represent the padding mode.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull (experimental).
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            norm_coords: whether to normalize the coordinates from `[-(size-1)/2, (size-1)/2]` to
                `[0, size - 1]` (for ``monai/csrc`` implementation) or
                `[-1, 1]` (for torch ``grid_sample`` implementation) to be compatible with the underlying
                resampling API.
            device: device on which the tensor will be allocated.
            dtype: data type for resampling computation. Defaults to ``float64`` for best precision.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always `float32`.

        """
        self.mode = mode
        self.padding_mode = padding_mode
        self.norm_coords = norm_coords
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        img: torch.Tensor,
        grid: torch.Tensor | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        dtype: DtypeLike = None,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
                if ``norm_coords`` is True, the grid values must be in `[-(size-1)/2, (size-1)/2]`.
                if ``USE_COMPILED=True`` and ``norm_coords=False``, grid values must be in `[0, size-1]`.
                if ``USE_COMPILED=False`` and ``norm_coords=False``, grid values must be in `[-1, 1]`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull (experimental).
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses an integer to represent the padding mode.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull (experimental).
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                To be compatible with other modules, the output data type is always `float32`.

        See also:
            :py:const:`monai.config.USE_COMPILED`
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if grid is None:
            return img
        _device = img.device if isinstance(img, torch.Tensor) else self.device
        _dtype = dtype or self.dtype or img.dtype
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=_dtype, device=_device)
        grid_t, *_ = convert_to_dst_type(grid, img_t, dtype=grid.dtype, wrap_sequence=True)
        grid_t = grid_t.clone(memory_format=torch.contiguous_format)

        if self.norm_coords:
            grid_t[-1] = where(grid_t[-1] != 0, grid_t[-1], 1.0)  # type: ignore
        sr = min(len(img_t.shape[1:]), 3)

        _interp_mode = self.mode if mode is None else mode
        _padding_mode = self.padding_mode if padding_mode is None else padding_mode
        if look_up_option(str(_interp_mode), SplineMode, default=None) is not None:
            self._backend = TransformBackends.NUMPY
        else:
            self._backend = TransformBackends.TORCH

        if USE_COMPILED or self._backend == TransformBackends.NUMPY:
            if self.norm_coords:
                for i, dim in enumerate(img_t.shape[1 : 1 + sr]):
                    grid_t[i] = (max(dim, 2) / 2.0 - 0.5 + grid_t[i]) / grid_t[-1:]
            grid_t = grid_t[:sr]
            if USE_COMPILED and self._backend == TransformBackends.TORCH:  # compiled is using torch backend param name
                grid_t = moveaxis(grid_t, 0, -1)  # type: ignore
                bound = 1 if _padding_mode == "reflection" else _padding_mode
                if _interp_mode == "bicubic":
                    interp = 3
                elif _interp_mode == "bilinear":
                    interp = 1
                else:
                    interp = GridSampleMode(_interp_mode)  # type: ignore
                out = grid_pull(
                    img_t.unsqueeze(0),
                    grid_t.unsqueeze(0).to(img_t),
                    bound=bound,
                    extrapolate=True,
                    interpolation=interp,
                )[0]
            elif self._backend == TransformBackends.NUMPY:
                is_cuda = img_t.is_cuda
                img_np = (convert_to_cupy if is_cuda else convert_to_numpy)(img_t, wrap_sequence=True)
                grid_np, *_ = convert_to_dst_type(grid_t, img_np, wrap_sequence=True)
                _map_coord = (cupy_ndi if is_cuda else np_ndi).map_coordinates
                out = (cupy if is_cuda else np).stack(
                    [
                        _map_coord(c, grid_np, order=int(_interp_mode), mode=look_up_option(_padding_mode, NdimageMode))
                        for c in img_np
                    ]
                )
                out = convert_to_dst_type(out, img_t)[0]
        else:
            if self.norm_coords:
                for i, dim in enumerate(img_t.shape[1 : 1 + sr]):
                    grid_t[i] = 2.0 / (max(2, dim) - 1.0) * grid_t[i] / grid_t[-1:]
            index_ordering: list[int] = list(range(sr - 1, -1, -1))
            grid_t = moveaxis(grid_t[index_ordering], 0, -1)  # type: ignore
            out = torch.nn.functional.grid_sample(
                img_t.unsqueeze(0),
                grid_t.unsqueeze(0).to(img_t),
                mode=GridSampleMode(_interp_mode),
                padding_mode=GridSamplePadMode(_padding_mode),
                align_corners=True,
            )[0]
        out_val, *_ = convert_to_dst_type(out, dst=img, dtype=np.float32)
        return out_val


class GridResampler(Transform):
    """
    Affine transforms on the coordinates.

    Args:
        rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
            Defaults to no rotation.
        shear_params: shearing factors for affine matrix, take a 3D affine as example::

            [
                [1.0, params[0], params[1], 0.0],
                [params[2], 1.0, params[3], 0.0],
                [params[4], params[5], 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

            a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
        translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
            pixel/voxel relative to the center of the input image. Defaults to no translation.
        scale_params: scale factor for every spatial dims. a tuple of 2 floats for 2D,
            a tuple of 3 floats for 3D. Defaults to `1.0`.
        dtype: data type for the grid computation. Defaults to ``float32``.
            If ``None``, use the data type of input data (if `grid` is provided).
        device: device on which the tensor will be allocated, if a new grid is generated.
        affine: If applied, ignore the params (`rotate_params`, etc.) and use the
            supplied matrix. Should be square with each side = num of image spatial
            dimensions + 1.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        rotate_params: Sequence[float] | float | None = None,
        shear_params: Sequence[float] | float | None = None,
        translate_params: Sequence[float] | float | None = None,
        scale_params: Sequence[float] | float | None = None,
        device: torch.device | None = None,
        dtype: DtypeLike = np.float32,
        affine: NdarrayOrTensor | None = None,
    ) -> None:
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params
        self.device = device
        self.dtype = dtype
        self.affine = affine

    def __call__(
        self, spatial_size: Sequence[int] | None = None, grid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The grid can be initialized with a `spatial_size` parameter, or provided directly as `grid`.
        Therefore, either `spatial_size` or `grid` must be provided.
        When initialising from `spatial_size`, the backend "torch" will be used.

        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Raises:
            ValueError: When ``grid=None`` and ``spatial_size=None``. Incompatible values.

        """
        if grid is None:  # create grid from spatial_size
            if spatial_size is None:
                raise ValueError("Incompatible values: grid=None and spatial_size=None.")
            grid_ = create_grid(spatial_size, device=self.device, backend="torch", dtype=self.dtype)
        else:
            grid_ = grid
        _dtype = self.dtype or grid_.dtype
        grid_: torch.Tensor = convert_to_tensor(grid_, dtype=_dtype, track_meta=get_track_meta())  # type: ignore
        _b = TransformBackends.TORCH
        _device = grid_.device  # type: ignore
        affine: NdarrayOrTensor
        # if self.affine is None:
        #     spatial_dims = len(grid_.shape) - 1
        #     affine = torch.eye(spatial_dims + 1, device=_device)
        #     if self.rotate_params:
        #         affine = affine @ create_rotate(spatial_dims, self.rotate_params, device=_device, backend=_b)
        #     if self.shear_params:
        #         affine = affine @ create_shear(spatial_dims, self.shear_params, device=_device, backend=_b)
        #     if self.translate_params:
        #         affine = affine @ create_translate(spatial_dims, self.translate_params, device=_device, backend=_b)
        #     if self.scale_params:
        #         affine = affine @ create_scale(spatial_dims, self.scale_params, device=_device, backend=_b)
        # else:
        #     affine = self.affine
        affine = self.affine

        affine = to_affine_nd(len(grid_) - 1, affine)
        affine = convert_to_tensor(affine, device=grid_.device, dtype=grid_.dtype, track_meta=False)  # type: ignore
        grid_ = (affine @ grid_.reshape((grid_.shape[0], -1))).reshape([-1] + list(grid_.shape[1:]))
        return grid_, affine  # type: ignore


class Resampler(InvertibleTransform):
    """
    TODO: refactor for lazy resampling
    Transform ``img`` given the affine parameters.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    """

    backend = list(set(GridResampler.backend) & set(ResampleImpl.backend))

    @deprecated_arg(name="norm_coords", since="0.8")
    def __init__(
        self,
        rotate_params: Sequence[float] | float | None = None,
        shear_params: Sequence[float] | float | None = None,
        translate_params: Sequence[float] | float | None = None,
        scale_params: Sequence[float] | float | None = None,
        affine: NdarrayOrTensor | None = None,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        normalized: bool = False,
        norm_coords: bool = True,
        device: torch.device | None = None,
        dtype: DtypeLike = np.float32,
        image_only: bool = False,
    ) -> None:
        """
        The affine transformations are applied in rotate, shear, translate, scale order.

        Args:
            rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
                Defaults to no rotation.
            shear_params: shearing factors for affine matrix, take a 3D affine as example::

                [
                    [1.0, params[0], params[1], 0.0],
                    [params[2], 1.0, params[3], 0.0],
                    [params[4], params[5], 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]

                a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
            translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
                pixel/voxel relative to the center of the input image. Defaults to no translation.
            scale_params: scale factor for every spatial dims. a tuple of 2 floats for 2D,
                a tuple of 3 floats for 3D. Defaults to `1.0`.
            affine: If applied, ignore the params (`rotate_params`, etc.) and use the
                supplied matrix. Should be square with each side = num of image spatial
                dimensions + 1.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            normalized: indicating whether the provided `affine` is defined to include a normalization
                transform converting the coordinates from `[-(size-1)/2, (size-1)/2]` (defined in ``create_grid``) to
                `[0, size - 1]` or `[-1, 1]` in order to be compatible with the underlying resampling API.
                If `normalized=False`, additional coordinate normalization will be applied before resampling.
                See also: :py:func:`monai.networks.utils.normalize_transform`.
            device: device on which the tensor will be allocated.
            dtype: data type for resampling computation. Defaults to ``float32``.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always `float32`.
            image_only: if True return only the image volume, otherwise return (image, affine).

        .. deprecated:: 0.8.1
            ``norm_coords`` is deprecated, please use ``normalized`` instead
            (the new flag is a negation, i.e., ``norm_coords == not normalized``).

        """
        self.affine_grid = GridResampler(
            rotate_params=rotate_params,
            shear_params=shear_params,
            translate_params=translate_params,
            scale_params=scale_params,
            affine=affine,
            dtype=dtype,
            device=device,
        )
        self.image_only = image_only
        self.norm_coord = not normalized
        self.resampler = ResampleImpl(norm_coords=self.norm_coord, device=device, dtype=dtype)
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode: str = padding_mode

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: Sequence[int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, NdarrayOrTensor]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_size = img.shape[1:]
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img_size)
        _mode = mode if mode is not None else self.mode
        _padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        grid, affine = self.affine_grid(spatial_size=sp_size)
        out = self.resampler(img, grid=grid, mode=_mode, padding_mode=_padding_mode)
        if not isinstance(out, MetaTensor):
            return out if self.image_only else (out, affine)
        if get_track_meta():
            out.meta = img.meta  # type: ignore
            self.update_meta(out, affine, img_size, sp_size)
            self.push_transform(
                out, orig_size=img_size, extra_info={"affine": affine, "mode": _mode, "padding_mode": _padding_mode}
            )
        return out if self.image_only else (out, affine)

    @classmethod
    def compute_w_affine(cls, affine, mat, img_size, sp_size):
        r = len(affine) - 1
        mat = to_affine_nd(r, mat)
        shift_1 = create_translate(r, [float(d - 1) / 2 for d in img_size[:r]])
        shift_2 = create_translate(r, [-float(d - 1) / 2 for d in sp_size[:r]])
        mat = shift_1 @ convert_data_type(mat, np.ndarray)[0] @ shift_2
        return affine @ convert_to_dst_type(mat, affine)[0]

    def update_meta(self, img, mat, img_size, sp_size):
        affine = convert_data_type(img.affine, torch.Tensor)[0]
        img.affine = Resampler.compute_w_affine(affine, mat, img_size, sp_size)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        orig_size = transform[TraceKeys.ORIG_SIZE]
        # Create inverse transform
        fwd_affine = transform[TraceKeys.EXTRA_INFO]["affine"]
        mode = transform[TraceKeys.EXTRA_INFO]["mode"]
        padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
        inv_affine = linalg_inv(fwd_affine)
        inv_affine = convert_to_dst_type(inv_affine, data, dtype=inv_affine.dtype)[0]

        affine_grid = GridResampler(affine=inv_affine)
        grid, _ = affine_grid(orig_size)
        # Apply inverse transform
        out = self.resampler(data, grid, mode, padding_mode)
        if not isinstance(out, MetaTensor):
            out = MetaTensor(out)
        out.meta = data.meta  # type: ignore
        self.update_meta(out, inv_affine, data.shape[1:], orig_size)
        return out

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

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

import monai
from monai.networks import to_norm_affine
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    convert_to_dst_type,
    ensure_tuple,
    look_up_option,
    optional_import,
)

_C, _ = optional_import("monai._C")

__all__ = ["AffineTransform", "grid_pull", "grid_push", "grid_count", "grid_grad"]


class _GridPull(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, interpolation, bound, extrapolate):
        opt = (bound, interpolation, extrapolate)
        output = _C.grid_pull(input, grid, *opt)
        if input.requires_grad or grid.requires_grad:
            ctx.opt = opt
            ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            return None, None, None, None, None
        var = ctx.saved_tensors
        opt = ctx.opt
        grads = _C.grid_pull_backward(grad, *var, *opt)
        if ctx.needs_input_grad[0]:
            return grads[0], grads[1] if ctx.needs_input_grad[1] else None, None, None, None
        if ctx.needs_input_grad[1]:
            return None, grads[0], None, None, None


def grid_pull(
    input: torch.Tensor, grid: torch.Tensor, interpolation="linear", bound="zero", extrapolate: bool = True
) -> torch.Tensor:
    """
    Sample an image with respect to a deformation field.

    `interpolation` can be an int, a string or an InterpolationType.
    Possible values are::

        - 0 or 'nearest'    or InterpolationType.nearest
        - 1 or 'linear'     or InterpolationType.linear
        - 2 or 'quadratic'  or InterpolationType.quadratic
        - 3 or 'cubic'      or InterpolationType.cubic
        - 4 or 'fourth'     or InterpolationType.fourth
        - 5 or 'fifth'      or InterpolationType.fifth
        - 6 or 'sixth'      or InterpolationType.sixth
        - 7 or 'seventh'    or InterpolationType.seventh

    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific interpolation orders.

    `bound` can be an int, a string or a BoundType.
    Possible values are::

        - 0 or 'replicate' or 'nearest'      or BoundType.replicate or 'border'
        - 1 or 'dct1'      or 'mirror'       or BoundType.dct1
        - 2 or 'dct2'      or 'reflect'      or BoundType.dct2
        - 3 or 'dst1'      or 'antimirror'   or BoundType.dst1
        - 4 or 'dst2'      or 'antireflect'  or BoundType.dst2
        - 5 or 'dft'       or 'wrap'         or BoundType.dft
        - 7 or 'zero'      or 'zeros'        or BoundType.zero

    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific boundary conditions.
    `sliding` is a specific condition than only applies to flow fields
    (with as many channels as dimensions). It cannot be dimension-specific.
    Note that:

        - `dft` corresponds to circular padding
        - `dct2` corresponds to Neumann boundary conditions (symmetric)
        - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)

    See Also:
        - https://en.wikipedia.org/wiki/Discrete_cosine_transform
        - https://en.wikipedia.org/wiki/Discrete_sine_transform
        - ``help(monai._C.BoundType)``
        - ``help(monai._C.InterpolationType)``

    Args:
        input: Input image. `(B, C, Wi, Hi, Di)`.
        grid: Deformation field. `(B, Wo, Ho, Do, 1|2|3)`.
        interpolation (int or list[int] , optional): Interpolation order.
            Defaults to `'linear'`.
        bound (BoundType, or list[BoundType], optional): Boundary conditions.
            Defaults to `'zero'`.
        extrapolate: Extrapolate out-of-bound data.
            Defaults to `True`.

    Returns:
        output (torch.Tensor): Deformed image `(B, C, Wo, Ho, Do)`.

    """
    # Convert parameters
    bound = [_C.BoundType.__members__[b] if isinstance(b, str) else _C.BoundType(b) for b in ensure_tuple(bound)]
    interpolation = [
        _C.InterpolationType.__members__[i] if isinstance(i, str) else _C.InterpolationType(i)
        for i in ensure_tuple(interpolation)
    ]
    out: torch.Tensor
    out = _GridPull.apply(input, grid, interpolation, bound, extrapolate)
    if isinstance(input, monai.data.MetaTensor):
        out = convert_to_dst_type(out, dst=input)[0]
    return out


class _GridPush(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, shape, interpolation, bound, extrapolate):
        opt = (bound, interpolation, extrapolate)
        output = _C.grid_push(input, grid, shape, *opt)
        if input.requires_grad or grid.requires_grad:
            ctx.opt = opt
            ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            return None, None, None, None, None, None
        var = ctx.saved_tensors
        opt = ctx.opt
        grads = _C.grid_push_backward(grad, *var, *opt)
        if ctx.needs_input_grad[0]:
            return grads[0], grads[1] if ctx.needs_input_grad[1] else None, None, None, None, None
        if ctx.needs_input_grad[1]:
            return None, grads[0], None, None, None, None


def grid_push(
    input: torch.Tensor, grid: torch.Tensor, shape=None, interpolation="linear", bound="zero", extrapolate: bool = True
):
    """
    Splat an image with respect to a deformation field (pull adjoint).

    `interpolation` can be an int, a string or an InterpolationType.
    Possible values are::

        - 0 or 'nearest'    or InterpolationType.nearest
        - 1 or 'linear'     or InterpolationType.linear
        - 2 or 'quadratic'  or InterpolationType.quadratic
        - 3 or 'cubic'      or InterpolationType.cubic
        - 4 or 'fourth'     or InterpolationType.fourth
        - 5 or 'fifth'      or InterpolationType.fifth
        - 6 or 'sixth'      or InterpolationType.sixth
        - 7 or 'seventh'    or InterpolationType.seventh

    A list of values can be provided, in the order `[W, H, D]`,
    to specify dimension-specific interpolation orders.

    `bound` can be an int, a string or a BoundType.
    Possible values are::

        - 0 or 'replicate' or 'nearest'      or BoundType.replicate
        - 1 or 'dct1'      or 'mirror'       or BoundType.dct1
        - 2 or 'dct2'      or 'reflect'      or BoundType.dct2
        - 3 or 'dst1'      or 'antimirror'   or BoundType.dst1
        - 4 or 'dst2'      or 'antireflect'  or BoundType.dst2
        - 5 or 'dft'       or 'wrap'         or BoundType.dft
        - 7 or 'zero'                        or BoundType.zero

    A list of values can be provided, in the order `[W, H, D]`,
    to specify dimension-specific boundary conditions.
    `sliding` is a specific condition than only applies to flow fields
    (with as many channels as dimensions). It cannot be dimension-specific.
    Note that:

        - `dft` corresponds to circular padding
        - `dct2` corresponds to Neumann boundary conditions (symmetric)
        - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)

    See Also:

        - https://en.wikipedia.org/wiki/Discrete_cosine_transform
        - https://en.wikipedia.org/wiki/Discrete_sine_transform
        - ``help(monai._C.BoundType)``
        - ``help(monai._C.InterpolationType)``

    Args:
        input: Input image `(B, C, Wi, Hi, Di)`.
        grid: Deformation field `(B, Wi, Hi, Di, 1|2|3)`.
        shape: Shape of the source image.
        interpolation (int or list[int] , optional): Interpolation order.
            Defaults to `'linear'`.
        bound (BoundType, or list[BoundType], optional): Boundary conditions.
            Defaults to `'zero'`.
        extrapolate: Extrapolate out-of-bound data.
            Defaults to `True`.

    Returns:
        output (torch.Tensor): Splatted image `(B, C, Wo, Ho, Do)`.

    """
    # Convert parameters
    bound = [_C.BoundType.__members__[b] if isinstance(b, str) else _C.BoundType(b) for b in ensure_tuple(bound)]
    interpolation = [
        _C.InterpolationType.__members__[i] if isinstance(i, str) else _C.InterpolationType(i)
        for i in ensure_tuple(interpolation)
    ]

    if shape is None:
        shape = tuple(input.shape[2:])

    out: torch.Tensor = _GridPush.apply(input, grid, shape, interpolation, bound, extrapolate)
    if isinstance(input, monai.data.MetaTensor):
        out = convert_to_dst_type(out, dst=input)[0]
    return out


class _GridCount(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, shape, interpolation, bound, extrapolate):
        opt = (bound, interpolation, extrapolate)
        output = _C.grid_count(grid, shape, *opt)
        if grid.requires_grad:
            ctx.opt = opt
            ctx.save_for_backward(grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        if ctx.needs_input_grad[0]:
            var = ctx.saved_tensors
            opt = ctx.opt
            return _C.grid_count_backward(grad, *var, *opt), None, None, None, None
        return None, None, None, None, None


def grid_count(grid: torch.Tensor, shape=None, interpolation="linear", bound="zero", extrapolate: bool = True):
    """
    Splatting weights with respect to a deformation field (pull adjoint).

    This function is equivalent to applying grid_push to an image of ones.

    `interpolation` can be an int, a string or an InterpolationType.
    Possible values are::

        - 0 or 'nearest'    or InterpolationType.nearest
        - 1 or 'linear'     or InterpolationType.linear
        - 2 or 'quadratic'  or InterpolationType.quadratic
        - 3 or 'cubic'      or InterpolationType.cubic
        - 4 or 'fourth'     or InterpolationType.fourth
        - 5 or 'fifth'      or InterpolationType.fifth
        - 6 or 'sixth'      or InterpolationType.sixth
        - 7 or 'seventh'    or InterpolationType.seventh

    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific interpolation orders.

    `bound` can be an int, a string or a BoundType.
    Possible values are::

        - 0 or 'replicate' or 'nearest'      or BoundType.replicate
        - 1 or 'dct1'      or 'mirror'       or BoundType.dct1
        - 2 or 'dct2'      or 'reflect'      or BoundType.dct2
        - 3 or 'dst1'      or 'antimirror'   or BoundType.dst1
        - 4 or 'dst2'      or 'antireflect'  or BoundType.dst2
        - 5 or 'dft'       or 'wrap'         or BoundType.dft
        - 7 or 'zero'                        or BoundType.zero

    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific boundary conditions.
    `sliding` is a specific condition than only applies to flow fields
    (with as many channels as dimensions). It cannot be dimension-specific.
    Note that:

        - `dft` corresponds to circular padding
        - `dct2` corresponds to Neumann boundary conditions (symmetric)
        - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)

    See Also:

        - https://en.wikipedia.org/wiki/Discrete_cosine_transform
        - https://en.wikipedia.org/wiki/Discrete_sine_transform
        - ``help(monai._C.BoundType)``
        - ``help(monai._C.InterpolationType)``

    Args:
        grid: Deformation field `(B, Wi, Hi, Di, 2|3)`.
        shape: shape of the source image.
        interpolation (int or list[int] , optional): Interpolation order.
            Defaults to `'linear'`.
        bound (BoundType, or list[BoundType], optional): Boundary conditions.
            Defaults to `'zero'`.
        extrapolate (bool, optional): Extrapolate out-of-bound data.
            Defaults to `True`.

    Returns:
        output (torch.Tensor): Splat weights `(B, 1, Wo, Ho, Do)`.

    """
    # Convert parameters
    bound = [_C.BoundType.__members__[b] if isinstance(b, str) else _C.BoundType(b) for b in ensure_tuple(bound)]
    interpolation = [
        _C.InterpolationType.__members__[i] if isinstance(i, str) else _C.InterpolationType(i)
        for i in ensure_tuple(interpolation)
    ]

    if shape is None:
        shape = tuple(grid.shape[2:])

    out: torch.Tensor = _GridCount.apply(grid, shape, interpolation, bound, extrapolate)
    if isinstance(input, monai.data.MetaTensor):
        out = convert_to_dst_type(out, dst=input)[0]
    return out


class _GridGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, interpolation, bound, extrapolate):
        opt = (bound, interpolation, extrapolate)
        output = _C.grid_grad(input, grid, *opt)
        if input.requires_grad or grid.requires_grad:
            ctx.opt = opt
            ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            return None, None, None, None, None
        var = ctx.saved_tensors
        opt = ctx.opt
        grads = _C.grid_grad_backward(grad, *var, *opt)
        if ctx.needs_input_grad[0]:
            return grads[0], grads[1] if ctx.needs_input_grad[1] else None, None, None, None
        if ctx.needs_input_grad[1]:
            return None, grads[0], None, None, None


def grid_grad(input: torch.Tensor, grid: torch.Tensor, interpolation="linear", bound="zero", extrapolate: bool = True):
    """
    Sample an image with respect to a deformation field.

    `interpolation` can be an int, a string or an InterpolationType.
    Possible values are::

        - 0 or 'nearest'    or InterpolationType.nearest
        - 1 or 'linear'     or InterpolationType.linear
        - 2 or 'quadratic'  or InterpolationType.quadratic
        - 3 or 'cubic'      or InterpolationType.cubic
        - 4 or 'fourth'     or InterpolationType.fourth
        - 5 or 'fifth'      or InterpolationType.fifth
        - 6 or 'sixth'      or InterpolationType.sixth
        - 7 or 'seventh'    or InterpolationType.seventh

    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific interpolation orders.

    `bound` can be an int, a string or a BoundType.
    Possible values are::

        - 0 or 'replicate' or 'nearest'      or BoundType.replicate
        - 1 or 'dct1'      or 'mirror'       or BoundType.dct1
        - 2 or 'dct2'      or 'reflect'      or BoundType.dct2
        - 3 or 'dst1'      or 'antimirror'   or BoundType.dst1
        - 4 or 'dst2'      or 'antireflect'  or BoundType.dst2
        - 5 or 'dft'       or 'wrap'         or BoundType.dft
        - 7 or 'zero'                        or BoundType.zero

    A list of values can be provided, in the order [W, H, D],
    to specify dimension-specific boundary conditions.
    `sliding` is a specific condition than only applies to flow fields
    (with as many channels as dimensions). It cannot be dimension-specific.
    Note that:

        - `dft` corresponds to circular padding
        - `dct2` corresponds to Neumann boundary conditions (symmetric)
        - `dst2` corresponds to Dirichlet boundary conditions (antisymmetric)

    See Also:

        - https://en.wikipedia.org/wiki/Discrete_cosine_transform
        - https://en.wikipedia.org/wiki/Discrete_sine_transform
        - ``help(monai._C.BoundType)``
        - ``help(monai._C.InterpolationType)``


    Args:
        input: Input image. `(B, C, Wi, Hi, Di)`.
        grid: Deformation field. `(B, Wo, Ho, Do, 2|3)`.
        interpolation (int or list[int] , optional): Interpolation order.
            Defaults to `'linear'`.
        bound (BoundType, or list[BoundType], optional): Boundary conditions.
            Defaults to `'zero'`.
        extrapolate: Extrapolate out-of-bound data. Defaults to `True`.

    Returns:
        output (torch.Tensor): Sampled gradients (B, C, Wo, Ho, Do, 1|2|3).

    """
    # Convert parameters
    bound = [_C.BoundType.__members__[b] if isinstance(b, str) else _C.BoundType(b) for b in ensure_tuple(bound)]
    interpolation = [
        _C.InterpolationType.__members__[i] if isinstance(i, str) else _C.InterpolationType(i)
        for i in ensure_tuple(interpolation)
    ]

    out: torch.Tensor = _GridGrad.apply(input, grid, interpolation, bound, extrapolate)
    if isinstance(input, monai.data.MetaTensor):
        out = convert_to_dst_type(out, dst=input)[0]
    return out


class AffineTransform(nn.Module):
    def __init__(
        self,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        normalized: bool = False,
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.ZEROS,
        align_corners: bool = False,
        reverse_indexing: bool = True,
        zero_centered: Optional[bool] = None,
    ) -> None:
        """
        Apply affine transformations with a batch of affine matrices.

        When `normalized=False` and `reverse_indexing=True`,
        it does the commonly used resampling in the 'pull' direction
        following the ``scipy.ndimage.affine_transform`` convention.
        In this case `theta` is equivalent to (ndim+1, ndim+1) input ``matrix`` of ``scipy.ndimage.affine_transform``,
        operates on homogeneous coordinates.
        See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html

        When `normalized=True` and `reverse_indexing=False`,
        it applies `theta` to the normalized coordinates (coords. in the range of [-1, 1]) directly.
        This is often used with `align_corners=False` to achieve resolution-agnostic resampling,
        thus useful as a part of trainable modules such as the spatial transformer networks.
        See also: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

        Args:
            spatial_size: output spatial shape, the full output shape will be
                `[N, C, *spatial_size]` where N and C are inferred from the `src` input of `self.forward`.
            normalized: indicating whether the provided affine matrix `theta` is defined
                for the normalized coordinates. If `normalized=False`, `theta` will be converted
                to operate on normalized coordinates as pytorch affine_grid works with the normalized
                coordinates.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"zeros"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: see also https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html.
            reverse_indexing: whether to reverse the spatial indexing of image and coordinates.
                set to `False` if `theta` follows pytorch's default "D, H, W" convention.
                set to `True` if `theta` follows `scipy.ndimage` default "i, j, k" convention.
            zero_centered: whether the affine is applied to coordinates in a zero-centered value range.
                With `zero_centered=True`, for example, the center of rotation will be the
                spatial center of the input; with `zero_centered=False`, the center of rotation will be the
                origin of the input. This option is only available when `normalized=False`,
                where the default behaviour is `False` if unspecified.
                See also: :py:func:`monai.networks.utils.normalize_transform`.
        """
        super().__init__()
        self.spatial_size = ensure_tuple(spatial_size) if spatial_size is not None else None
        self.normalized = normalized
        self.mode: str = look_up_option(mode, GridSampleMode)
        self.padding_mode: str = look_up_option(padding_mode, GridSamplePadMode)
        self.align_corners = align_corners
        self.reverse_indexing = reverse_indexing
        if zero_centered is not None and self.normalized:
            raise ValueError("`normalized=True` is not compatible with the `zero_centered` option.")
        self.zero_centered = zero_centered if zero_centered is not None else False

    def forward(
        self, src: torch.Tensor, theta: torch.Tensor, spatial_size: Optional[Union[Sequence[int], int]] = None
    ) -> torch.Tensor:
        """
        ``theta`` must be an affine transformation matrix with shape
        3x3 or Nx3x3 or Nx2x3 or 2x3 for spatial 2D transforms,
        4x4 or Nx4x4 or Nx3x4 or 3x4 for spatial 3D transforms,
        where `N` is the batch size. `theta` will be converted into float Tensor for the computation.

        Args:
            src (array_like): image in spatial 2D or 3D (N, C, spatial_dims),
                where N is the batch dim, C is the number of channels.
            theta (array_like): Nx3x3, Nx2x3, 3x3, 2x3 for spatial 2D inputs,
                Nx4x4, Nx3x4, 3x4, 4x4 for spatial 3D inputs. When the batch dimension is omitted,
                `theta` will be repeated N times, N is the batch dim of `src`.
            spatial_size: output spatial shape, the full output shape will be
                `[N, C, *spatial_size]` where N and C are inferred from the `src`.

        Raises:
            TypeError: When ``theta`` is not a ``torch.Tensor``.
            ValueError: When ``theta`` is not one of [Nxdxd, dxd].
            ValueError: When ``theta`` is not one of [Nx3x3, Nx4x4].
            TypeError: When ``src`` is not a ``torch.Tensor``.
            ValueError: When ``src`` spatially is not one of [2D, 3D].
            ValueError: When affine and image batch dimension differ.

        """
        # validate `theta`
        if not isinstance(theta, torch.Tensor):
            raise TypeError(f"theta must be torch.Tensor but is {type(theta).__name__}.")
        if theta.dim() not in (2, 3):
            raise ValueError(f"theta must be Nxdxd or dxd, got {theta.shape}.")
        if theta.dim() == 2:
            theta = theta[None]  # adds a batch dim.
        theta = theta.clone()  # no in-place change of theta
        theta_shape = tuple(theta.shape[1:])
        if theta_shape in ((2, 3), (3, 4)):  # needs padding to dxd
            pad_affine = torch.tensor([0, 0, 1] if theta_shape[0] == 2 else [0, 0, 0, 1])
            pad_affine = pad_affine.repeat(theta.shape[0], 1, 1).to(theta)
            pad_affine.requires_grad = False
            theta = torch.cat([theta, pad_affine], dim=1)
        if tuple(theta.shape[1:]) not in ((3, 3), (4, 4)):
            raise ValueError(f"theta must be Nx3x3 or Nx4x4, got {theta.shape}.")

        # validate `src`
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"src must be torch.Tensor but is {type(src).__name__}.")
        sr = src.dim() - 2  # input spatial rank
        if sr not in (2, 3):
            raise ValueError(f"Unsupported src dimension: {sr}, available options are [2, 3].")

        # set output shape
        src_size = tuple(src.shape)
        dst_size = src_size  # default to the src shape
        if self.spatial_size is not None:
            dst_size = src_size[:2] + self.spatial_size
        if spatial_size is not None:
            dst_size = src_size[:2] + ensure_tuple(spatial_size)

        # reverse and normalize theta if needed
        if not self.normalized:
            theta = to_norm_affine(
                affine=theta,
                src_size=src_size[2:],
                dst_size=dst_size[2:],
                align_corners=self.align_corners,
                zero_centered=self.zero_centered,
            )
        if self.reverse_indexing:
            rev_idx = torch.as_tensor(range(sr - 1, -1, -1), device=src.device)
            theta[:, :sr] = theta[:, rev_idx]
            theta[:, :, :sr] = theta[:, :, rev_idx]
        if (theta.shape[0] == 1) and src_size[0] > 1:
            # adds a batch dim to `theta` in order to match `src`
            theta = theta.repeat(src_size[0], 1, 1)
        if theta.shape[0] != src_size[0]:
            raise ValueError(
                f"affine and image batch dimension must match, got affine={theta.shape[0]} image={src_size[0]}."
            )

        grid = nn.functional.affine_grid(theta=theta[:, :sr], size=list(dst_size), align_corners=self.align_corners)
        dst = nn.functional.grid_sample(
            input=src.contiguous(),
            grid=grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return dst

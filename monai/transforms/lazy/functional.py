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

from __future__ import annotations

from typing import Any

import torch

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import to_affine_nd
from monai.transforms.lazy.utils import (
    affine_from_pending,
    combine_transforms,
    is_compatible_apply_kwargs,
    kwargs_from_pending,
    resample,
)
from monai.utils import LazyAttr, look_up_option

__all__ = ["apply_transforms"]

__override_keywords = {"mode", "padding_mode", "dtype", "align_corners", "resample_mode", "device"}


def apply_transforms(
    data: torch.Tensor | MetaTensor, pending: list | None = None, overrides: dict | None = None, **kwargs: Any
):
    """
    This method applies pending transforms to `data` tensors.
    Currently, only 2d and 3d input are supported.

    Args:
        data: A torch Tensor or a monai MetaTensor.
        pending: pending transforms. This must be set if data is a Tensor, but is optional if data is a MetaTensor.
        overrides: a dictionary of overrides for the transform arguments. The keys must be one of:

            - mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order ``0-5`` (integers).
                Interpolation mode to calculate output values. Defaults to None.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's `an integer`, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            - padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to None.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            - dtype: data type for resampling computation. Defaults to ``float64``.
                If ``None``, use the data type of input data, this option may not be compatible the resampling backend.
            - align_corners: Geometrically, we consider the pixels of the input as squares rather than points, when using
                the PyTorch resampling backend. Defaults to ``False``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            - device: device for resampling computation. Defaults to ``None``.
            - resample_mode: the mode of resampling, currently support ``"auto"``. Setting to other values will use the
                :py:class:`monai.transforms.SpatialResample` for resampling (instead of potentially crop/pad).

    """
    overrides = (overrides or {}).copy()
    overrides.update((kwargs or {}).copy())
    for k in overrides:
        look_up_option(k, __override_keywords)  # check existence of the key

    if isinstance(data, MetaTensor) and pending is None:
        pending = data.pending_operations.copy()
        data.clear_pending_operations()
    pending = [] if pending is None else pending

    if not pending:
        return data, []

    cumulative_xform = affine_from_pending(pending[0])
    if cumulative_xform.shape[0] == 3:
        cumulative_xform = to_affine_nd(3, cumulative_xform)

    cur_kwargs = kwargs_from_pending(pending[0])
    override_kwargs: dict[str, Any] = {}
    if "mode" in overrides:
        override_kwargs[LazyAttr.INTERP_MODE] = overrides["mode"]
    if "padding_mode" in overrides:
        override_kwargs[LazyAttr.PADDING_MODE] = overrides["padding_mode"]
    if "align_corners" in overrides:
        override_kwargs[LazyAttr.ALIGN_CORNERS] = overrides["align_corners"]
    if "resample_mode" in overrides:
        override_kwargs[LazyAttr.RESAMPLE_MODE] = overrides["resample_mode"]
    override_dtype = overrides.get("dtype", torch.float64)
    override_kwargs[LazyAttr.DTYPE] = data.dtype if override_dtype is None else override_dtype
    device = overrides.get("device")

    for p in pending[1:]:
        new_kwargs = kwargs_from_pending(p)
        if not is_compatible_apply_kwargs(cur_kwargs, new_kwargs):
            # carry out an intermediate resample here due to incompatibility between arguments
            _cur_kwargs = cur_kwargs.copy()
            _cur_kwargs.update(override_kwargs)
            data = resample(data.to(device), cumulative_xform, _cur_kwargs)
        next_matrix = affine_from_pending(p)
        if next_matrix.shape[0] == 3:
            next_matrix = to_affine_nd(3, next_matrix)
        cumulative_xform = combine_transforms(cumulative_xform, next_matrix)
        cur_kwargs.update(new_kwargs)
    cur_kwargs.update(override_kwargs)
    data = resample(data.to(device), cumulative_xform, cur_kwargs)
    if isinstance(data, MetaTensor):
        for p in pending:
            data.push_applied_operation(p)
    return data, pending

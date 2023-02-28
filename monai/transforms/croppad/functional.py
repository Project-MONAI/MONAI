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
"""
A collection of "functional" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.nn.functional import pad as pad_pt

from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.inverse import TraceableTransform
from monai.transforms.utils import convert_pad_mode, create_translate
from monai.utils import TraceKeys, convert_to_dst_type, convert_to_tensor, ensure_tuple

__all__ = ["pad_nd", "pad_func", "crop_func"]


def _np_pad(img: torch.Tensor, pad_width: list[tuple[int, int]], mode: str, **kwargs) -> torch.Tensor:
    if isinstance(img, torch.Tensor):
        if img.is_cuda:
            warnings.warn(f"Padding: moving img {img.shape} from cuda to cpu for dtype={img.dtype} mode={mode}.")
        img_np = img.detach().cpu().numpy()
    else:
        img_np = img
    mode = convert_pad_mode(dst=img_np, mode=mode).value
    if mode == "constant" and "value" in kwargs:
        kwargs["constant_values"] = kwargs.pop("value")
    out = torch.as_tensor(np.pad(img, pad_width, mode=mode, **kwargs))  # type: ignore
    if isinstance(img, MetaTensor):
        out = convert_to_dst_type(out, dst=img)[0]
    return out


def _pt_pad(img: torch.Tensor, pad_width: list[tuple[int, int]], mode: str, **kwargs) -> torch.Tensor:
    mode = convert_pad_mode(dst=img, mode=mode).value
    if mode == "constant" and "constant_values" in kwargs:
        _kwargs = kwargs.copy()
        _kwargs["value"] = _kwargs.pop("constant_values")
    else:
        _kwargs = kwargs
    pt_pad_width = [val for sublist in pad_width[1:] for val in sublist[::-1]][::-1]
    # torch.pad expects `[B, C, H, W, [D]]` shape
    return pad_pt(img.unsqueeze(0), pt_pad_width, mode=mode, **_kwargs).squeeze(0)


def pad_nd(img: torch.Tensor, to_pad: list[tuple[int, int]], mode: str, **kwargs):
    """
    PyTorch/Numpy pad ``img`` with integers ``to_pad`` amounts. Depending on the ``mode`` and input dtype,
    a suitable backend will be used automatically.

    Args:
        img: data to be transformed, assuming `img` is channel-first and padding doesn't apply to the channel dim.
        to_pad: the amount to be padded in each dimension [(low_H, high_H), (low_W, high_W), ...].
            default to `self.to_pad`.
        mode: available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """
    if mode in {"linear_ramp", "maximum", "mean", "median", "minimum", "symmetric", "empty"}:
        return _np_pad(img, pad_width=to_pad, mode=mode, **kwargs)
    mode = convert_pad_mode(dst=img, mode=mode).value
    try:
        _pad = (
            _np_pad
            if mode in {"reflect", "replicate"} and img.dtype in {torch.int16, torch.int64, torch.bool, torch.uint8}
            else _pt_pad
        )
        return _pad(img, pad_width=to_pad, mode=mode, **kwargs)
    except (ValueError, TypeError, RuntimeError) as err:
        if isinstance(err, NotImplementedError) or any(
            k in str(err) for k in ("supported", "unexpected keyword", "implemented", "value")
        ):
            return _np_pad(img, pad_width=to_pad, mode=mode, **kwargs)
        raise ValueError(f"{img.shape} {to_pad} {mode} {kwargs} {img.dtype} {img.device}") from err


def pad_func(img: torch.Tensor, to_pad: list[tuple[int, int]], mode: str, transform_info: dict, kwargs):
    """
    Functional implementation of padding a MetaTensor. This function operates eagerly or lazily according
    to ``transform_info[TraceKeys.LAZY_EVALUATION]`` (default ``False``).

    Args:
        img: data to be transformed, assuming `img` is channel-first and padding doesn't apply to the channel dim.
        to_pad: the amount to be padded in each dimension [(low_H, high_H), (low_W, high_W), ...].
            note that it including channel dimension.
        mode: available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """
    extra_info = {"padded": to_pad}
    img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    spatial_rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else 3
    do_pad = np.asarray(to_pad).any()
    if do_pad:
        to_pad = list(to_pad)
        if len(to_pad) < len(img.shape):
            to_pad = list(to_pad) + [(0, 0)] * (len(img.shape) - len(to_pad))
        to_shift = [-s[0] for s in to_pad[1:]]  # skipping the channel pad
        xform = create_translate(spatial_rank, to_shift)
        shape = [d + s + e for d, (s, e) in zip(img_size, to_pad[1:])]
    else:
        shape = img_size
        xform = torch.eye(int(spatial_rank) + 1, device=torch.device("cpu"), dtype=torch.float64)
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=shape,
        affine=xform,
        extra_info=extra_info,
        orig_size=img_size,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    out = pad_nd(out, to_pad, mode, **kwargs) if do_pad else out
    out = convert_to_tensor(out, track_meta=get_track_meta())
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out


def crop_func(img: torch.Tensor, slices: tuple[slice, ...], transform_info: dict):
    """
    Functional implementation of cropping a MetaTensor. This function operates eagerly or lazily according
    to ``transform_info[TraceKeys.LAZY_EVALUATION]`` (default ``False``).

    Args:
        img: data to be transformed, assuming `img` is channel-first and cropping doesn't apply to the channel dim.
        slices: the crop slices computed based on specified `center & size` or `start & end` or `slices`.
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    spatial_rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else 3
    cropped = np.asarray([[s.indices(o)[0], o - s.indices(o)[1]] for s, o in zip(slices[1:], img_size)])
    extra_info = {"cropped": cropped.flatten().tolist()}
    to_shift = []
    for i, s in enumerate(ensure_tuple(slices)[1:]):
        if s.start is not None:
            to_shift.append(img_size[i] + s.start if s.start < 0 else s.start)
        else:
            to_shift.append(0)
    shape = [s.indices(o)[1] - s.indices(o)[0] for s, o in zip(slices[1:], img_size)]
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=shape,
        affine=create_translate(spatial_rank, to_shift),
        extra_info=extra_info,
        orig_size=img_size,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    out = out[slices]
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out

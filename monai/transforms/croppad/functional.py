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
A collection of "functional" transforms for spatial operations.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.nn.functional import pad as pad_pt

from monai.config.type_definitions import NdarrayTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import to_affine_nd
from monai.transforms.inverse import TraceableTransform
from monai.transforms.utils import convert_pad_mode, create_translate
from monai.utils import PytorchPadMode, convert_to_dst_type, convert_to_numpy, convert_to_tensor, ensure_tuple

__all__ = ["pad_nd", "pad_func", "crop_func", "crop_or_pad_nd"]


def _convert_pt_pad_mode(padding_mode):
    """get the most similar mode of `pad` from ``padding_mode`` of the spatial resampling."""
    if padding_mode is None or padding_mode in ("zeros", "constant", "grid-constant"):
        return PytorchPadMode.CONSTANT
    elif padding_mode in ("reflection", "reflect", "mirror", "grid-mirror"):
        return PytorchPadMode.REFLECT
    elif padding_mode in ("wrap", "grid-wrap"):
        return PytorchPadMode.CIRCULAR
    return PytorchPadMode.REPLICATE  # "nearest", "border", and others


def _np_pad(img: NdarrayTensor, pad_width: list[tuple[int, int]], mode: str, **kwargs) -> NdarrayTensor:
    if isinstance(img, torch.Tensor):
        if img.is_cuda:
            warnings.warn(f"Padding: moving img {img.shape} from cuda to cpu for dtype={img.dtype} mode={mode}.")
        img_np = img.detach().cpu().numpy()
    else:
        img_np = img
    mode = convert_pad_mode(dst=img_np, mode=mode).value
    if mode == "constant" and "value" in kwargs:
        kwargs["constant_values"] = kwargs.pop("value")
    img_np = np.pad(img_np, pad_width, mode=mode, **kwargs)  # type: ignore
    return convert_to_dst_type(img_np, dst=img)[0]


def _pt_pad(img: NdarrayTensor, pad_width: list[tuple[int, int]], mode: str, **kwargs) -> NdarrayTensor:
    img_pt = torch.as_tensor(img)
    mode = convert_pad_mode(dst=img_pt, mode=mode).value
    if mode == "constant" and "constant_values" in kwargs:
        _kwargs = kwargs.copy()
        _kwargs["value"] = _kwargs.pop("constant_values")
    else:
        _kwargs = kwargs
    pt_pad_width = [val for sublist in pad_width[1:] for val in sublist[::-1]][::-1]
    # torch.pad expects `[B, C, H, W, [D]]` shape
    img_pt = pad_pt(img_pt.unsqueeze(0), pt_pad_width, mode=mode, **_kwargs).squeeze(0)
    return convert_to_dst_type(img_pt, dst=img)[0]


def pad_nd(
    img: NdarrayTensor, to_pad: list[tuple[int, int]], mode: str = PytorchPadMode.CONSTANT, **kwargs
) -> NdarrayTensor:
    """
    Pad `img` for a given an amount of padding in each dimension.

    `torch.nn.functional.pad` is used unless the mode or kwargs are not available in torch,
    in which case `np.pad` will be used.

    Args:
        img: data to be transformed, assuming `img` is channel-first and padding doesn't apply to the channel dim.
        to_pad: the amount to be padded in each dimension [(low_H, high_H), (low_W, high_W), ...].
            default to `self.to_pad`.
        mode: available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """
    if mode in {"linear_ramp", "maximum", "mean", "median", "minimum", "symmetric", "empty"}:
        return _np_pad(img, pad_width=to_pad, mode=mode, **kwargs)
    try:
        _pad = _np_pad
        if mode in {"constant", "reflect", "edge", "replicate", "wrap", "circular"} and img.dtype not in {
            torch.int16,
            torch.int64,
            torch.bool,
            torch.uint8,
        }:
            _pad = _pt_pad
        return _pad(img, pad_width=to_pad, mode=mode, **kwargs)
    except (ValueError, TypeError, RuntimeError) as err:
        if isinstance(err, NotImplementedError) or any(
            k in str(err) for k in ("supported", "unexpected keyword", "implemented", "value")
        ):
            return _np_pad(img, pad_width=to_pad, mode=mode, **kwargs)
        raise ValueError(
            f"{img.shape} {to_pad} {mode} {kwargs} {img.dtype} {img.device if isinstance(img, torch.Tensor) else None}"
        ) from err


def crop_or_pad_nd(img: torch.Tensor, translation_mat, spatial_size: tuple[int, ...], mode: str, **kwargs):
    """
    Crop or pad using the translation matrix and spatial size. The translation coefficients are rounded
    to the nearest integers. For a more generic implementation, please see :py:class:`monai.transforms.SpatialResample`.

    Args:
        img: data to be transformed, assuming `img` is channel-first and padding doesn't apply to the channel dim.
        translation_mat: the translation matrix to be applied to the image. A translation matrix generated by,
            for example, :py:func:`monai.transforms.utils.create_translate`. The translation coefficients are rounded
            to the nearest integers.
        spatial_size: the spatial size of the output image.
        mode: the padding mode.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
    """
    ndim = len(img.shape) - 1
    matrix_np = np.round(to_affine_nd(ndim, convert_to_numpy(translation_mat, wrap_sequence=True).copy()))
    matrix_np = to_affine_nd(len(spatial_size), matrix_np)
    cc = np.asarray(np.meshgrid(*[[0.5, x - 0.5] for x in spatial_size], indexing="ij"))
    cc = cc.reshape((len(spatial_size), -1))
    src_cc = np.floor(matrix_np @ np.concatenate((cc, np.ones_like(cc[:1]))))
    src_start, src_end = src_cc.min(axis=1), src_cc.max(axis=1)
    to_pad, to_crop, do_pad, do_crop = [(0, 0)], [slice(None)], False, False
    for s, e, sp in zip(src_start, src_end, img.shape[1:]):
        do_pad, do_crop = do_pad or s < 0 or e > sp - 1, do_crop or s > 0 or e < sp - 1
        to_pad += [(0 if s >= 0 else int(-s), 0 if e < sp - 1 else int(e - sp + 1))]
        to_crop += [slice(int(max(s, 0)), int(e + 1 + to_pad[-1][0]))]
    if do_pad:
        _mode = _convert_pt_pad_mode(mode)
        img = pad_nd(img, to_pad, mode=_mode, **kwargs)
    if do_crop:
        img = img[to_crop]
    return img


def pad_func(
    img: torch.Tensor,
    to_pad: tuple[tuple[int, int]],
    transform_info: dict,
    mode: str = PytorchPadMode.CONSTANT,
    lazy: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Functional implementation of padding a MetaTensor. This function operates eagerly or lazily according
    to ``lazy`` (default ``False``).

    `torch.nn.functional.pad` is used unless the mode or kwargs are not available in torch,
    in which case `np.pad` will be used.

    Args:
        img: data to be transformed, assuming `img` is channel-first and padding doesn't apply to the channel dim.
        to_pad: the amount to be padded in each dimension [(low_H, high_H), (low_W, high_W), ...].
            note that it including channel dimension.
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
        mode: available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        lazy: a flag indicating whether the operation should be performed in a lazy fashion or not.
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """
    extra_info = {"padded": to_pad, "mode": f"{mode}"}
    img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    spatial_rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else 3
    do_pad = np.asarray(to_pad).any()
    if do_pad:
        to_pad_list = [(int(p[0]), int(p[1])) for p in to_pad]
        if len(to_pad_list) < len(img.shape):
            to_pad_list += [(0, 0)] * (len(img.shape) - len(to_pad_list))
        to_shift = [-s[0] for s in to_pad_list[1:]]  # skipping the channel pad
        xform = create_translate(spatial_rank, to_shift)
        shape = [d + s + e for d, (s, e) in zip(img_size, to_pad_list[1:])]
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
        lazy=lazy,
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if lazy:
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info  # type: ignore
    out = pad_nd(out, to_pad_list, mode, **kwargs) if do_pad else out
    out = convert_to_tensor(out, track_meta=get_track_meta())
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out  # type: ignore


def crop_func(img: torch.Tensor, slices: tuple[slice, ...], lazy: bool, transform_info: dict) -> torch.Tensor:
    """
    Functional implementation of cropping a MetaTensor. This function operates eagerly or lazily according
    to ``lazy`` (default ``False``).

    Args:
        img: data to be transformed, assuming `img` is channel-first and cropping doesn't apply to the channel dim.
        slices: the crop slices computed based on specified `center & size` or `start & end` or `slices`.
        lazy: a flag indicating whether the operation should be performed in a lazy fashion or not.
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
        lazy=lazy,
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if lazy:
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info  # type: ignore
    out = out[slices]
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out  # type: ignore

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

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    optional_import,
)

tqdm, _ = optional_import("tqdm", name="tqdm")
_nearest_mode = "nearest-exact"

__all__ = ["sliding_window_inference", "sliding_window_inference_with_reduction"]


def sliding_window_inference(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
    roi_weight_map: torch.Tensor | None = None,
    process_fn: Callable | None = None,
    buffer_steps: int | None = None,
    buffer_dim: int = -1,
    with_coord: bool = False,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    """
    Sliding window inference on `inputs` with `predictor`.

    The outputs of `predictor` could be a tensor, a tuple, or a dictionary of tensors.
    Each output in the tuple or dict value is allowed to have different resolutions with respect to the input.
    e.g., the input patch spatial size is [128,128,128], the output (a tuple of two patches) patch sizes
    could be ([128,64,256], [64,32,128]).
    In this case, the parameter `overlap` and `roi_size` need to be carefully chosen to ensure the output ROI is still
    an integer. If the predictor's input and output spatial sizes are not equal, we recommend choosing the parameters
    so that `overlap*roi_size*output_size/input_size` is an integer (for each spatial dimension).

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences, this must be a single value or a tuple with values
            for each spatial dimension (eg. 2 for 2D, 3 for 3D).
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            The outputs of the function call ``predictor(patch_data)`` should be a tensor, a tuple, or a dictionary
            with Tensor values. Each output in the tuple or dict value should have the same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch's spatial size, M is the number of output channels,
            N is `sw_batch_size`, e.g., the input shape is (7, 1, 128,128,128),
            the output could be a tuple of two tensors, with shapes: ((7, 5, 128, 64, 256), (7, 4, 64, 32, 128)).
            In this case, the parameter `overlap` and `roi_size` need to be carefully chosen
            to ensure the scaled output ROI sizes are still integers.
            If the `predictor`'s input and output spatial sizes are different,
            we recommend choosing the parameters so that ``overlap*roi_size*zoom_scale`` is an integer for each dimension.
        overlap: Amount of overlap between scans along each spatial dimension, defaults to ``0.25``.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a `tqdm` progress bar.
        roi_weight_map: pre-computed (non-negative) weight map for each ROI.
            If not given, and ``mode`` is not `constant`, this map will be computed on the fly.
        process_fn: process inference output and adjust the importance map per window
        buffer_steps: the number of sliding window iterations along the ``buffer_dim``
            to be buffered on ``sw_device`` before writing to ``device``.
            (Typically, ``sw_device`` is ``cuda`` and ``device`` is ``cpu``.)
            default is None, no buffering. For the buffer dim, when spatial size is divisible by buffer_steps*roi_size,
            (i.e. no overlapping among the buffers) non_blocking copy may be automatically enabled for efficiency.
        buffer_dim: the spatial dimension along which the buffers are created.
            0 indicates the first spatial dimension. Default is -1, the last spatial dimension.
        with_coord: whether to pass the window coordinates to ``predictor``. Default is False.
            If True, the signature of ``predictor`` should be ``predictor(patch_data, patch_coord, ...)``.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - Inputs must be channel-first and have a batch dim (NCHW / NCDHW).
        - If your data is NHWC/NDHWC, please apply `EnsureChannelFirst` / `EnsureChannelFirstd` upstream.

    Raises:
        ValueError: When the input dimensions do not match the expected dimensions based on ``roi_size``.

    """
    num_spatial_dims = len(inputs.shape) - 2

    # Only perform strict shape validation if roi_size is a sequence (explicit dimensions).
    # If roi_size is an integer, it is broadcast to all dimensions, so we cannot
    # infer the expected dimensionality to enforce a strict check here.
    if isinstance(roi_size, Sequence):
        roi_dims = len(roi_size)
        if num_spatial_dims != roi_dims:
            raise ValueError(
                f"Inputs must have {roi_dims + 2} dimensions for {roi_dims}D roi_size "
                f"(Batch, Channel, {', '.join(['Spatial'] * roi_dims)}), "
                f"but got inputs shape {inputs.shape}.\n"
                "If you have channel-last data (e.g. B, D, H, W, C), please use "
                "monai.transforms.EnsureChannelFirst or EnsureChannelFirstd upstream."
            )
    # -----------------------------------------------------------------
    buffered = buffer_steps is not None and buffer_steps > 0
    if buffered:
        if buffer_dim < -num_spatial_dims or buffer_dim > num_spatial_dims:
            raise ValueError(f"buffer_dim must be in [{-num_spatial_dims}, {num_spatial_dims}], got {buffer_dim}.")
        if buffer_dim < 0:
            buffer_dim += num_spatial_dims
    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device

    condition = kwargs.pop("condition", None)

    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    roi_size = fall_back_tuple(roi_size, image_size_)

    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)
        if condition is not None:
            condition = F.pad(condition, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)

    # Store all slices
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=not buffered)

    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    windows_range: Iterable
    if not buffered:
        non_blocking = False
        windows_range = range(0, total_slices, sw_batch_size)
    else:
        slices, n_per_batch, b_slices, windows_range = _create_buffered_slices(
            slices, batch_size, sw_batch_size, buffer_dim, buffer_steps
        )
        non_blocking, _ss = torch.cuda.is_available(), -1
        for x in b_slices[:n_per_batch]:
            if x[1] < _ss:  # detect overlapping slices
                non_blocking = False
                break
            _ss = x[2]

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map_ = roi_weight_map
    else:
        try:
            valid_p_size = ensure_tuple(valid_patch_size)
            importance_map_ = compute_importance_map(
                valid_p_size, mode=mode, sigma_scale=sigma_scale, device=sw_device, dtype=compute_dtype
            )
            if len(importance_map_.shape) == num_spatial_dims and not process_fn:
                importance_map_ = importance_map_[None, None]  # adds batch, channel dimensions
        except Exception as e:
            raise RuntimeError(
                f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map_ = convert_data_type(importance_map_, torch.Tensor, device=sw_device, dtype=compute_dtype)[0]

    # stores output and count map
    output_image_list, count_map_list, sw_device_buffer, b_s, b_i = [], [], [], 0, 0  # type: ignore
    # for each patch
    for slice_g in tqdm(windows_range) if progress else windows_range:
        slice_range = range(slice_g, min(slice_g + sw_batch_size, b_slices[b_s][0] if buffered else total_slices))
        unravel_slice = [
            [slice(idx // num_win, idx // num_win + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        if sw_batch_size > 1:
            win_data = torch.cat([inputs[ensure_tuple(win_slice)] for win_slice in unravel_slice]).to(sw_device)
            if condition is not None:
                win_condition = torch.cat([condition[ensure_tuple(win_slice)] for win_slice in unravel_slice]).to(
                    sw_device
                )
                kwargs["condition"] = win_condition
        else:
            s0 = unravel_slice[0]
            s0_idx = ensure_tuple(s0)

            win_data = inputs[s0_idx].to(sw_device)
            if condition is not None:
                win_condition = condition[s0_idx].to(sw_device)
                kwargs["condition"] = win_condition

        if with_coord:
            seg_prob_out = predictor(win_data, unravel_slice, *args, **kwargs)
        else:
            seg_prob_out = predictor(win_data, *args, **kwargs)
        # convert seg_prob_out to tuple seg_tuple, this does not allocate new memory.
        dict_keys, seg_tuple = _flatten_struct(seg_prob_out)
        if process_fn:
            seg_tuple, w_t = process_fn(seg_tuple, win_data, importance_map_)
        else:
            w_t = importance_map_
        if len(w_t.shape) == num_spatial_dims:
            w_t = w_t[None, None]
        w_t = w_t.to(dtype=compute_dtype, device=sw_device)
        if buffered:
            c_start, c_end = b_slices[b_s][1:]
            if not sw_device_buffer:
                k = seg_tuple[0].shape[1]  # len(seg_tuple) > 1 is currently ignored
                sp_size = list(image_size)
                sp_size[buffer_dim] = c_end - c_start
                sw_device_buffer = [torch.zeros(size=[1, k, *sp_size], dtype=compute_dtype, device=sw_device)]
            for p, s in zip(seg_tuple[0], unravel_slice):
                offset = s[buffer_dim + 2].start - c_start
                s[buffer_dim + 2] = slice(offset, offset + roi_size[buffer_dim])
                s[0] = slice(0, 1)
                sw_device_buffer[0][ensure_tuple(s)] += p * w_t
            b_i += len(unravel_slice)
            if b_i < b_slices[b_s][0]:
                continue
        else:
            sw_device_buffer = list(seg_tuple)

        for ss in range(len(sw_device_buffer)):
            b_shape = sw_device_buffer[ss].shape
            seg_chns, seg_shape = b_shape[1], b_shape[2:]
            z_scale = None
            if not buffered and seg_shape != roi_size:
                z_scale = [out_w_i / float(in_w_i) for out_w_i, in_w_i in zip(seg_shape, roi_size)]
                w_t = F.interpolate(w_t, seg_shape, mode=_nearest_mode)
            if len(output_image_list) <= ss:
                output_shape = [batch_size, seg_chns]
                output_shape += [int(_i * _z) for _i, _z in zip(image_size, z_scale)] if z_scale else list(image_size)
                # allocate memory to store the full output and the count for overlapping parts
                new_tensor: Callable = torch.empty if non_blocking else torch.zeros  # type: ignore
                output_image_list.append(new_tensor(output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                w_t_ = w_t.to(device)
                for __s in slices:
                    if z_scale is not None:
                        __s = tuple(slice(int(_si.start * z_s), int(_si.stop * z_s)) for _si, z_s in zip(__s, z_scale))
                    count_map_list[-1][(slice(None), slice(None), *__s)] += w_t_
            if buffered:
                o_slice = [slice(None)] * len(inputs.shape)
                o_slice[buffer_dim + 2] = slice(c_start, c_end)
                img_b = b_s // n_per_batch  # image batch index
                o_slice[0] = slice(img_b, img_b + 1)
                o_slice_idx = ensure_tuple(o_slice)
                if non_blocking:
                    output_image_list[0][o_slice_idx].copy_(sw_device_buffer[0], non_blocking=non_blocking)
                else:
                    output_image_list[0][o_slice_idx] += sw_device_buffer[0].to(device=device)
            else:
                sw_device_buffer[ss] *= w_t
                sw_device_buffer[ss] = sw_device_buffer[ss].to(device)
                _compute_coords(unravel_slice, z_scale, output_image_list[ss], sw_device_buffer[ss])
        sw_device_buffer = []
        if buffered:
            b_s += 1

    if non_blocking:
        torch.cuda.current_stream().synchronize()

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] /= count_map_list.pop(0)

    # remove padding if image_size smaller than roi_size
    if any(pad_size):
        kwargs.update({"pad_size": pad_size})
        for ss, output_i in enumerate(output_image_list):
            zoom_scale = [_shape_d / _roi_size_d for _shape_d, _roi_size_d in zip(output_i.shape[2:], roi_size)]
            final_slicing: list[slice] = []
            for sp in range(num_spatial_dims):
                si = num_spatial_dims - sp - 1
                slice_dim = slice(
                    int(round(pad_size[sp * 2] * zoom_scale[si])),
                    int(round((pad_size[sp * 2] + image_size_[si]) * zoom_scale[si])),
                )
                final_slicing.insert(0, slice_dim)
            output_image_list[ss] = output_i[(slice(None), slice(None), *final_slicing)]

    final_output = _pack_struct(output_image_list, dict_keys)
    if temp_meta is not None:
        final_output = convert_to_dst_type(final_output, temp_meta, device=device)[0]
    else:
        final_output = convert_to_dst_type(final_output, inputs, device=device)[0]

    return final_output  # type: ignore


def _create_buffered_slices(slices, batch_size, sw_batch_size, buffer_dim, buffer_steps):
    """rearrange slices for buffering"""
    slices_np = np.asarray(slices)
    slices_np = slices_np[np.argsort(slices_np[:, buffer_dim, 0], kind="mergesort")]
    slices = [tuple(slice(c[0], c[1]) for c in i) for i in slices_np]
    slices_np = slices_np[:, buffer_dim]

    _, _, _b_lens = np.unique(slices_np[:, 0], return_counts=True, return_index=True)
    b_ends = np.cumsum(_b_lens).tolist()  # possible buffer flush boundaries
    x = [0, *b_ends][:: min(len(b_ends), int(buffer_steps))]
    if x[-1] < b_ends[-1]:
        x.append(b_ends[-1])
    n_per_batch = len(x) - 1
    windows_range = [
        range(b * x[-1] + x[i], b * x[-1] + x[i + 1], sw_batch_size)
        for b in range(batch_size)
        for i in range(n_per_batch)
    ]
    b_slices = []
    for _s, _r in enumerate(windows_range):
        s_s = slices_np[windows_range[_s - 1].stop % len(slices) if _s > 0 else 0, 0]
        s_e = slices_np[(_r.stop - 1) % len(slices), 1]
        b_slices.append((_r.stop, s_s, s_e))  # buffer index, slice start, slice end
    windows_range = itertools.chain(*windows_range)  # type: ignore
    return slices, n_per_batch, b_slices, windows_range


def _compute_coords(coords, z_scale, out, patch):
    """sliding window batch spatial scaling indexing for multi-resolution outputs."""
    for original_idx, p in zip(coords, patch):
        idx_zm = list(original_idx)  # 4D for 2D image, 5D for 3D image
        if z_scale:
            for axis in range(2, len(idx_zm)):
                idx_zm[axis] = slice(
                    int(original_idx[axis].start * z_scale[axis - 2]), int(original_idx[axis].stop * z_scale[axis - 2])
                )
        out[ensure_tuple(idx_zm)] += p


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: Sequence[float]
) -> tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError(f"len(image_size) {len(image_size)} different from spatial dims {num_spatial_dims}.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError(f"len(roi_size) {len(roi_size)} different from spatial dims {num_spatial_dims}.")

    scan_interval = []
    for i, o in zip(range(num_spatial_dims), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def _flatten_struct(seg_out):
    dict_keys = None
    seg_probs: tuple[torch.Tensor, ...]
    if isinstance(seg_out, torch.Tensor):
        seg_probs = (seg_out,)
    elif isinstance(seg_out, Mapping):
        dict_keys = sorted(seg_out.keys())  # track predictor's output keys
        seg_probs = tuple(seg_out[k] for k in dict_keys)
    else:
        seg_probs = ensure_tuple(seg_out)
    return dict_keys, seg_probs


def _pack_struct(seg_out, dict_keys=None):
    if dict_keys is not None:
        return dict(zip(dict_keys, seg_out))
    if isinstance(seg_out, (list, tuple)) and len(seg_out) == 1:
        return seg_out[0]
    return ensure_tuple(seg_out)


def sliding_window_inference_with_reduction(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: Sequence[float] | float = 0.25,
    mode: BlendMode | str = BlendMode.CONSTANT,
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
    roi_weight_map: torch.Tensor | None = None,
    reduction_fn: Callable | None = None,
    reduction_dim: int = 1,
    output_dtype: torch.dtype | None = None,
    outer_dim: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Memory-efficient sliding window inference with online reduction.

    Instead of accumulating full-resolution probabilities for the entire volume,
    this function processes slabs along one spatial dimension and applies a
    reduction function (e.g., ``torch.argmax``) as soon as each slab is complete.
    This dramatically reduces peak memory for many-class segmentation tasks.

    For example, 100-class segmentation on a 384-cubed volume with roi_size=128:

    - Standard SWI: ~22 GB peak (B x 100 x 384^3 float32 buffer)
    - This function: ~7.5 GB peak (B x 100 x 128 x 384^2 slab buffer + B x 1 x 384^3 uint8 output)

    Note:
        This function only supports single-tensor predictor output (not tuple/dict).
        Multi-resolution outputs (where model output spatial size differs from ``roi_size``)
        are not supported. For those cases, use ``sliding_window_inference``.

    Args:
        inputs: input image to be processed (assuming NCHW[D]).
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            ``predictor(patch_data)`` should return a single tensor of shape NM'H'W'[D'],
            where H'W'[D'] must equal the input patch spatial size.
        overlap: Amount of overlap between scans along each spatial dimension, defaults to ``0.25``.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.
        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs.
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data and slab accumulation buffer.
            By default the device of the ``inputs`` is used.
        device: device for the final reduced output tensor.
            By default the device of the ``inputs`` is used.
        progress: whether to print a `tqdm` progress bar.
        roi_weight_map: pre-computed (non-negative) weight map for each ROI.
        reduction_fn: function to reduce the accumulated probabilities per slab.
            Signature: ``reduction_fn(tensor, dim) -> tensor``.
            Default: ``lambda x, dim: torch.argmax(x, dim=dim)``.
        reduction_dim: the dimension to reduce over. Default: 1 (channel dimension).
        output_dtype: dtype for the reduced output tensor. Default: ``torch.uint8``.
        outer_dim: spatial dimension index (0-based) to iterate slabs along.
            Default: automatically selects the largest spatial dimension.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Returns:
        Reduced output tensor. For the default ``argmax`` with ``reduction_dim=1``,
        the output shape is ``(B, 1, *spatial)`` with dtype ``output_dtype``.

    """
    if reduction_fn is None:
        reduction_fn = lambda x, dim: torch.argmax(x, dim=dim)
    if output_dtype is None:
        output_dtype = torch.uint8

    num_spatial_dims = len(inputs.shape) - 2
    if isinstance(roi_size, Sequence):
        if num_spatial_dims != len(roi_size):
            raise ValueError(
                f"Inputs must have {len(roi_size) + 2} dimensions for {len(roi_size)}D roi_size "
                f"(Batch, Channel, {', '.join(['Spatial'] * len(roi_size))}), "
                f"but got inputs shape {inputs.shape}."
            )

    overlap = ensure_tuple_rep(overlap, num_spatial_dims)
    for o in overlap:
        if o < 0 or o >= 1:
            raise ValueError(f"overlap must be >= 0 and < 1, got {overlap}.")
    compute_dtype = inputs.dtype

    batch_size, _, *image_size_ = inputs.shape
    device = device or inputs.device
    sw_device = sw_device or inputs.device

    temp_meta = None
    if isinstance(inputs, MetaTensor):
        temp_meta = MetaTensor([]).copy_meta_from(inputs, copy_attr=False)
    inputs = convert_data_type(inputs, torch.Tensor, wrap_sequence=True)[0]
    roi_size = fall_back_tuple(roi_size, image_size_)

    # pad if image is smaller than roi
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)

    # auto-select outer_dim as the largest spatial dimension
    if outer_dim is None:
        outer_dim = max(range(num_spatial_dims), key=lambda d: image_size[d])
    elif outer_dim < 0:
        outer_dim += num_spatial_dims
    if not (0 <= outer_dim < num_spatial_dims):
        raise ValueError(f"outer_dim must be in [0, {num_spatial_dims}), got {outer_dim}.")

    roi_d = roi_size[outer_dim]

    # scan intervals and all patch positions
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    all_slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=True)

    # importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and roi_weight_map is not None:
        importance_map = roi_weight_map
    else:
        try:
            valid_p_size = ensure_tuple(valid_patch_size)
            importance_map = compute_importance_map(
                valid_p_size, mode=mode, sigma_scale=sigma_scale, device=sw_device, dtype=compute_dtype
            )
            if len(importance_map.shape) == num_spatial_dims:
                importance_map = importance_map[None, None]
        except Exception as e:
            raise RuntimeError(
                f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}\n"
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map = convert_data_type(importance_map, torch.Tensor, device=sw_device, dtype=compute_dtype)[0]

    # group slices by outer_dim start position
    slices_by_outer: dict[int, list] = {}
    for s in all_slices:
        key = s[outer_dim].start
        if key not in slices_by_outer:
            slices_by_outer[key] = []
        slices_by_outer[key].append(s)
    outer_starts = sorted(slices_by_outer.keys())

    ndim_full = 2 + num_spatial_dims
    slab_buffer: torch.Tensor | None = None
    count_buffer: torch.Tensor | None = None
    output: torch.Tensor | None = None
    buf_start = 0

    def _make_outer_idx(outer_slice, batch_s=slice(None), channel_s=slice(None)):
        """Build index tuple with a specific slice along outer_dim, slice(None) elsewhere."""
        idx: list[slice] = [batch_s, channel_s]
        for d in range(num_spatial_dims):
            idx.append(outer_slice if d == outer_dim else slice(None))
        return tuple(idx)

    def _flush_rows(n_rows):
        """Reduce completed rows from the slab buffer and store in the output."""
        nonlocal buf_start, slab_buffer, count_buffer
        if n_rows <= 0 or slab_buffer is None or count_buffer is None or output is None:
            return

        flush_idx = _make_outer_idx(slice(0, n_rows))
        count_flush_idx = _make_outer_idx(slice(0, n_rows), channel_s=slice(0, 1))

        completed = slab_buffer[flush_idx] / count_buffer[count_flush_idx].clamp_(min=1e-8)
        reduced = reduction_fn(completed.to(device), reduction_dim)
        if reduced.dim() < ndim_full:
            reduced = reduced.unsqueeze(reduction_dim)

        out_idx = _make_outer_idx(slice(buf_start, buf_start + n_rows))
        output[out_idx] = reduced.to(dtype=output_dtype)

        remaining = roi_d - n_rows
        if remaining > 0:
            src_idx = _make_outer_idx(slice(n_rows, roi_d))
            dst_idx = _make_outer_idx(slice(0, remaining))
            count_src = _make_outer_idx(slice(n_rows, roi_d), channel_s=slice(0, 1))
            count_dst = _make_outer_idx(slice(0, remaining), channel_s=slice(0, 1))
            overlap_data = slab_buffer[src_idx].clone()
            overlap_count = count_buffer[count_src].clone()
            slab_buffer.zero_()
            count_buffer.zero_()
            slab_buffer[dst_idx] = overlap_data
            count_buffer[count_dst] = overlap_count
        else:
            slab_buffer.zero_()
            count_buffer.zero_()

        buf_start += n_rows

    # main loop over slab positions
    windows_iter = tqdm(outer_starts) if progress else outer_starts
    for p in windows_iter:
        # flush rows completed before this outer position
        _flush_rows(p - buf_start)

        patches = slices_by_outer[p]
        all_windows = [(b, s) for b in range(batch_size) for s in patches]

        for chunk_start in range(0, len(all_windows), sw_batch_size):
            chunk = all_windows[chunk_start : chunk_start + sw_batch_size]

            win_data = torch.cat(
                [inputs[(slice(b, b + 1), slice(None)) + tuple(s)] for b, s in chunk], dim=0
            ).to(sw_device)

            seg_prob = predictor(win_data, *args, **kwargs)
            if not isinstance(seg_prob, torch.Tensor):
                raise TypeError(
                    "sliding_window_inference_with_reduction only supports single-tensor predictor output, "
                    f"got {type(seg_prob)}. Use sliding_window_inference for tuple/dict outputs."
                )

            # initialize buffers on first prediction
            if slab_buffer is None:
                seg_shape = tuple(seg_prob.shape[2:])
                if seg_shape != tuple(roi_size):
                    raise NotImplementedError(
                        f"Model output spatial size {seg_shape} differs from roi_size {tuple(roi_size)}. "
                        "Multi-resolution output is not supported by "
                        "sliding_window_inference_with_reduction."
                    )
                n_ch = seg_prob.shape[1]
                buf_shape = [batch_size, n_ch]
                count_shape = [batch_size, 1]
                for d in range(num_spatial_dims):
                    dim_size = roi_d if d == outer_dim else image_size[d]
                    buf_shape.append(dim_size)
                    count_shape.append(dim_size)
                slab_buffer = torch.zeros(buf_shape, dtype=compute_dtype, device=sw_device)
                count_buffer = torch.zeros(count_shape, dtype=compute_dtype, device=sw_device)

                # determine output shape after reduction
                dummy = torch.zeros([1, n_ch] + [1] * num_spatial_dims)
                dummy_reduced = reduction_fn(dummy, reduction_dim)
                if dummy_reduced.dim() < ndim_full:
                    dummy_reduced = dummy_reduced.unsqueeze(reduction_dim)
                out_ch = dummy_reduced.shape[1]
                output = torch.zeros([batch_size, out_ch] + list(image_size), dtype=output_dtype, device=device)

            # accumulate into slab buffer
            w_t = importance_map
            for i, (b, s) in enumerate(chunk):
                local_outer = s[outer_dim].start - buf_start
                local_s = list(s)
                local_s[outer_dim] = slice(local_outer, local_outer + roi_d)
                buf_idx = (slice(b, b + 1), slice(None)) + tuple(local_s)
                count_idx = (slice(b, b + 1), slice(0, 1)) + tuple(local_s)

                slab_buffer[buf_idx] += seg_prob[i : i + 1].to(dtype=compute_dtype, device=sw_device) * w_t
                count_buffer[count_idx] += w_t

    # flush remaining rows
    if slab_buffer is not None:
        _flush_rows(image_size[outer_dim] - buf_start)

    if output is None:
        raise RuntimeError("No windows were processed. Check roi_size and image dimensions.")

    # remove padding
    if any(pad_size):
        final_slicing: list[slice] = [slice(None), slice(None)]
        for sp in range(num_spatial_dims):
            si = num_spatial_dims - sp - 1
            final_slicing.insert(2, slice(pad_size[sp * 2], pad_size[sp * 2] + image_size_[si]))
        output = output[tuple(final_slicing)]

    if temp_meta is not None:
        output = convert_to_dst_type(output, temp_meta, device=device)[0]

    return output

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

__all__ = ["sliding_window_inference"]


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
    streaming_slab: bool = False,
    slab_dim: int = 0,
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
        streaming_slab: whether to use active slab streaming for constant O(1) memory along ``slab_dim``.
            Defaults to False. When True, only an active slab of depth ``roi_size[slab_dim]`` is maintained in
            ``sw_device`` memory, and finalized slices are continuously normalized and streamed to ``device``.
            To prevent CUDA memory exhaustion when inferring massive volumes, set ``device="cpu"`` so the
            full output tensor resides in host RAM while active window computation remains on ``sw_device``.
            Streaming may group windows differently from ordinary inference, changing results for
            predictors whose output depends on other windows in the batch.
        slab_dim: the spatial dimension along which streaming slab progresses (default is 0).
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
    if streaming_slab:
        if buffered:
            raise ValueError("Cannot specify both `buffer_steps` and `streaming_slab=True`.")
        if slab_dim < -num_spatial_dims or slab_dim >= num_spatial_dims:
            raise ValueError(f"slab_dim must be in [{-num_spatial_dims}, {num_spatial_dims - 1}], got {slab_dim}.")
        if slab_dim < 0:
            slab_dim += num_spatial_dims
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

    if streaming_slab:
        return _sliding_window_inference_streaming_slab(
            inputs,
            roi_size,
            sw_batch_size,
            predictor,
            overlap,
            mode,
            sigma_scale,
            padding_mode,
            cval,
            sw_device,
            device,
            progress,
            roi_weight_map,
            process_fn,
            slab_dim,
            with_coord,
            image_size,
            image_size_,
            pad_size,
            temp_meta,
            condition,
            args=args,
            kwargs=kwargs,
        )

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


def _sliding_window_inference_streaming_slab(
    inputs: torch.Tensor,
    roi_size: Sequence[int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
    overlap: Sequence[float],
    mode: BlendMode | str,
    sigma_scale: Sequence[float] | float,
    padding_mode: PytorchPadMode | str,
    cval: float,
    sw_device: torch.device | str,
    device: torch.device | str,
    progress: bool,
    roi_weight_map: torch.Tensor | None,
    process_fn: Callable | None,
    slab_dim: int,
    with_coord: bool,
    image_size: Sequence[int],
    image_size_: Sequence[int],
    pad_size: list[int],
    temp_meta: Any,
    condition: torch.Tensor | None,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
    """Stream sliding window inference with slab-sized accumulation buffers.

    Each output tensor covers the full output volume on ``device``, while only an active
    slab of size ``roi_size[slab_dim]`` is maintained in ``sw_device`` accumulation memory.

    Args:
        inputs: model input data for inference.
        roi_size: the spatial window size of patches.
        sw_batch_size: the batch size to run window slices.
        predictor: target model callable to execute inference.
        overlap: amount of overlap between scans along each spatial dimension.
        mode: how to blend output of overlapping windows.
        sigma_scale: standard deviation coefficient of Gaussian window.
        padding_mode: padding mode when ``roi_size`` is larger than inputs.
        cval: fill value for constant padding mode.
        sw_device: device for window computation.
        device: device for stitched output prediction.
        progress: whether to print a tqdm progress bar.
        roi_weight_map: precomputed weight map for ROI patches.
        process_fn: optional callable to process inference output and adjust importance map per window.
        slab_dim: spatial dimension along which streaming progresses.
        with_coord: whether to pass window coordinates to predictor.
        image_size: input spatial dimensions after optional padding.
        image_size_: original input spatial dimensions.
        pad_size: spatial padding amounts.
        temp_meta: metadata for output conversion.
        condition: optional condition tensor matching inputs.
        args: optional positional arguments passed to ``predictor``.
        kwargs: optional keyword arguments passed to ``predictor``.

    Returns:
        torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
            stitched prediction covering the full volume on ``device``.
    """
    num_spatial_dims = len(image_size)
    batch_size = inputs.shape[0]
    compute_dtype = inputs.dtype
    pred_kwargs = kwargs.copy() if kwargs else {}

    # Store all slices and order along slab_dim
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval, return_slice=True)
    if slab_dim != 0:
        slices = sorted(
            slices, key=lambda s: (s[slab_dim].start, *(s[d].start for d in range(num_spatial_dims) if d != slab_dim))
        )

    # Window importance map
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
                importance_map_ = importance_map_[None, None]
        except Exception as e:
            raise RuntimeError(
                f"patch size {valid_p_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map_ = convert_data_type(importance_map_, torch.Tensor, device=sw_device, dtype=compute_dtype)[0]
    if len(importance_map_.shape) == num_spatial_dims and not process_fn:
        importance_map_ = importance_map_[None, None]

    # Group slices by start coordinate along slab_dim
    groups: dict[int, list[tuple[slice, ...]]] = {}
    for s in slices:
        z_start = int(s[slab_dim].start)
        if z_start not in groups:
            groups[z_start] = []
        groups[z_start].append(s)
    unique_starts = sorted(groups.keys())

    p_slab = roi_size[slab_dim]
    output_image_list: list[torch.Tensor] = []
    active_val: list[torch.Tensor] = []
    active_count_list: list[torch.Tensor] = []
    dict_keys: tuple | list | None = None
    z_scales: list[list[float] | None] = []
    scaled_p_slabs: list[int] = []
    initialized = False

    total_steps = sum((len(groups[z]) + sw_batch_size - 1) // sw_batch_size for z in unique_starts) * batch_size
    pbar = tqdm(total=total_steps) if progress and tqdm else None

    for b in range(batch_size):
        for i, z_start in enumerate(unique_starts):
            grp_slices = groups[z_start]
            for c_idx in range(0, len(grp_slices), sw_batch_size):
                chunk = grp_slices[c_idx : c_idx + sw_batch_size]
                unravel_slice = [[slice(b, b + 1), slice(None)] + list(s) for s in chunk]
                if len(chunk) > 1:
                    win_data = torch.cat([inputs[ensure_tuple(s)] for s in unravel_slice]).to(sw_device)
                    if condition is not None:
                        pred_kwargs["condition"] = torch.cat([condition[ensure_tuple(s)] for s in unravel_slice]).to(
                            sw_device
                        )
                else:
                    s0_idx = ensure_tuple(unravel_slice[0])
                    win_data = inputs[s0_idx].to(sw_device)
                    if condition is not None:
                        pred_kwargs["condition"] = condition[s0_idx].to(sw_device)

                if with_coord:
                    seg_prob_out = predictor(win_data, unravel_slice, *args, **pred_kwargs)
                else:
                    seg_prob_out = predictor(win_data, *args, **pred_kwargs)

                dict_keys, seg_tuple = _flatten_struct(seg_prob_out)
                if process_fn:
                    seg_tuple, w_t = process_fn(seg_tuple, win_data, importance_map_)
                else:
                    w_t = importance_map_

                if len(w_t.shape) == num_spatial_dims:
                    w_t = w_t[None, None]
                w_t = w_t.to(dtype=compute_dtype, device=sw_device)

                if not initialized:
                    initialized = True
                    if len(seg_tuple) == 0:
                        output_image_list = []
                        active_val = []
                        active_count_list = []
                    else:
                        for ss in range(len(seg_tuple)):
                            b_shape = seg_tuple[ss].shape
                            seg_chns, seg_shape = b_shape[1], b_shape[2:]
                            z_s = (
                                [out_w_i / float(in_w_i) for out_w_i, in_w_i in zip(seg_shape, roi_size)]
                                if seg_shape != roi_size
                                else None
                            )
                            z_scales.append(z_s)
                            output_shape = [batch_size, seg_chns]
                            output_shape += (
                                [int(_i * _z) for _i, _z in zip(image_size, z_s)] if z_s else list(image_size)
                            )
                            output_image_list.append(
                                torch.full(output_shape, float("nan"), dtype=compute_dtype, device=device)
                            )

                            active_slab_shape = [1, seg_chns]
                            active_slab_shape += (
                                [int(_i * _z) for _i, _z in zip(image_size, z_s)] if z_s else list(image_size)
                            )
                            sp_slab = int(p_slab * z_s[slab_dim]) if z_s else p_slab
                            scaled_p_slabs.append(sp_slab)
                            active_slab_shape[slab_dim + 2] = sp_slab
                            active_val.append(torch.zeros(active_slab_shape, dtype=compute_dtype, device=sw_device))

                            count_slab_shape = [1, 1] + [active_slab_shape[2 + d] for d in range(num_spatial_dims)]
                            active_count_list.append(
                                torch.zeros(count_slab_shape, dtype=compute_dtype, device=sw_device)
                            )

                if active_count_list:
                    for ss in range(len(seg_tuple)):
                        z_s = z_scales[ss]
                        sp_slab = scaled_p_slabs[ss]
                        head_shape = seg_tuple[ss].shape[2:]
                        if head_shape != w_t.shape[2:]:
                            head_w_t = F.interpolate(w_t, head_shape, mode=_nearest_mode)
                        else:
                            head_w_t = w_t

                        for p_idx, s in enumerate(chunk):
                            local_val_s = [slice(0, 1), slice(None)] + [slice(None)] * num_spatial_dims
                            local_count_s = [slice(0, 1), slice(None)] + [slice(None)] * num_spatial_dims
                            for d in range(num_spatial_dims):
                                if d == slab_dim:
                                    local_val_s[d + 2] = slice(0, sp_slab)
                                    local_count_s[d + 2] = slice(0, sp_slab)
                                else:
                                    st = int(s[d].start * z_s[d]) if z_s else s[d].start
                                    sp = int(s[d].stop * z_s[d]) if z_s else s[d].stop
                                    local_val_s[d + 2] = slice(st, sp)
                                    local_count_s[d + 2] = slice(st, sp)

                            window_w = head_w_t if head_w_t.shape[0] == 1 else head_w_t[p_idx : p_idx + 1]
                            patch_w = window_w.to(device=sw_device, dtype=compute_dtype)
                            patch_pred = (seg_tuple[ss][p_idx : p_idx + 1] * window_w).to(
                                device=sw_device, dtype=compute_dtype
                            )
                            active_val[ss][ensure_tuple(local_val_s)] += patch_pred
                            active_count_list[ss][ensure_tuple(local_count_s)] += patch_w

                if pbar is not None:
                    pbar.update(1)

            if active_count_list:
                next_z = unique_starts[i + 1] if i + 1 < len(unique_starts) else image_size[slab_dim]
                for ss in range(len(active_val)):
                    z_s = z_scales[ss]
                    sp_slab = scaled_p_slabs[ss]
                    scaled_z_start = int(z_start * z_s[slab_dim]) if z_s else z_start
                    scaled_next_z = int(next_z * z_s[slab_dim]) if z_s else next_z
                    scaled_num_retire = scaled_next_z - scaled_z_start

                    retire_val_slice = [slice(0, 1), slice(None)] + [slice(None)] * num_spatial_dims
                    retire_val_slice[slab_dim + 2] = slice(0, scaled_num_retire)
                    retire_count_slice = [slice(0, 1), slice(None)] + [slice(None)] * num_spatial_dims
                    retire_count_slice[slab_dim + 2] = slice(0, scaled_num_retire)

                    dst_slice = [slice(b, b + 1), slice(None)] + [slice(None)] * num_spatial_dims
                    dst_slice[slab_dim + 2] = slice(scaled_z_start, scaled_next_z)

                    denom = active_count_list[ss][ensure_tuple(retire_count_slice)].clone()
                    retire_slice_data = active_val[ss][ensure_tuple(retire_val_slice)] / denom
                    output_image_list[ss][ensure_tuple(dst_slice)] = retire_slice_data.to(device=device)

                    rem = sp_slab - scaled_num_retire
                    if rem > 0 and i + 1 < len(unique_starts):
                        shift_step = scaled_num_retire
                        src_shift = [slice(None)] * (num_spatial_dims + 2)
                        src_shift[slab_dim + 2] = slice(shift_step, shift_step + rem)

                        dst_shift = [slice(None)] * (num_spatial_dims + 2)
                        dst_shift[slab_dim + 2] = slice(0, rem)

                        clear_shift = [slice(None)] * (num_spatial_dims + 2)
                        clear_shift[slab_dim + 2] = slice(rem, sp_slab)

                        active_val[ss][ensure_tuple(dst_shift)] = active_val[ss][ensure_tuple(src_shift)].clone()
                        active_val[ss][ensure_tuple(clear_shift)].zero_()

                        active_count_list[ss][ensure_tuple(dst_shift)] = active_count_list[ss][
                            ensure_tuple(src_shift)
                        ].clone()
                        active_count_list[ss][ensure_tuple(clear_shift)].zero_()
                    else:
                        active_val[ss].zero_()
                        active_count_list[ss].zero_()

    if pbar is not None:
        pbar.close()

    if (torch.device(sw_device).type == "cuda" or torch.device(device).type == "cuda") and torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()

    if any(pad_size):
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

    return final_output

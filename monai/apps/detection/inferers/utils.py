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

from copy import deepcopy
from typing import Any, Callable, Dict, List, Sequence, Union

import torch
import torch.nn.functional as F

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.inferers.utils import _get_scan_interval
from monai.utils import BlendMode, PytorchPadMode, ensure_tuple, fall_back_tuple, look_up_option, optional_import

tqdm, _ = optional_import("tqdm", name="tqdm")

__all__ = ["sliding_window_inference_multioutput"]


def sliding_window_inference_multioutput(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    importance_map_roi_size: Union[torch.Tensor, None] = None,
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]:
    """
    Sliding window inference on `inputs` with a multi-output `predictor`.
    The outputs of `predictor` should either be Sequence with same length, or Dict with torch.Tensor values ans same keys.
    Each output in the Sequence or Dict value is allowed to have different resolutions with the input.
    e.g., the input size patch size is [128,128,128], the output patch sizes could be [128, 64, 256], [64, 32, 128].
    The sliding window ensemble is performed on the zoomed grids. But please make sure the scale is int.
    Also, `overlap*roi_size` should be divisible with the int scale for all the axis.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D],
            The outputs of the `predictor(patch_data)` should be a Sequence of Tensors or a Dict with Tensor values.
            Each output in the Sequence or Dict value has same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch spatial size, M is the number of output channels, N is `sw_batch_size`,
            e.g., the input shape is (7, 1, 128,128,128), the output shape could be (7, 5, 128, 64, 256), (7, 4, 64, 32, 128).
            The sliding window ensemble is performed on the zoomed grids.
            But please make sure the scale is int, i.e., `s = H/H'` should be int.
            Also, `overlap*roi_size` should be divisible with `s` for all the axis.
        overlap: Amount of overlap between scans.
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
        importance_map_roi_size: importance_map_roi_size computed for roi_size.
            If not given, this func will compute importance_map on the fly.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (importance_map_roi_size is not None):
        importance_map = importance_map_roi_size.to(compute_dtype).to(device)
    else:
        try:
            importance_map = compute_importance_map(
                valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device
            ).to(compute_dtype)
        except Exception as e:
            raise RuntimeError(
                "Seems to be OOM. Please try to use smaller patch size or use mode='constant' instead of mode='gaussian'. "
            ) from e
    # importance_map cannot be 0, otherwise we may end up with nans!
    min_non_zero = importance_map[importance_map != 0].min().item()
    importance_map[importance_map < min_non_zero] = min_non_zero  # to prevent NaN

    # Perform predictions
    dict_key = None
    output_image_list, count_map_list = [], []
    _initialized_list = [False]

    # for each patch
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        seg_prob_out = predictor(window_data, *args, **kwargs)  # batched patch segmentation

        # convert seg_prob_out to tuple seg_prob_tuple, this does not allocate new memory.
        if isinstance(seg_prob_out, torch.Tensor):
            seg_prob_tuple = (seg_prob_out,)
            tensor_output = True
        elif isinstance(seg_prob_out, dict):
            dict_key = list(seg_prob_out.keys())
            dict_key.sort()
            seg_prob_tuple = tuple(seg_prob_out[k] for k in dict_key)
            tensor_output = False
        else:
            seg_prob_tuple = ensure_tuple(seg_prob_out)
            tensor_output = False

        # for each output in multi-output list
        for ss in range(len(seg_prob_tuple)):
            seg_prob = seg_prob_tuple[ss].to(device)  # BxCxMxNxP or BxCxMxN

            # compute zoom scal: out_roi_size/in_roi_size
            zoom_scale = []
            for axis in range(num_spatial_dims):
                scale = seg_prob.shape[2 + axis] / float(window_data.shape[2 + axis])
                if not (image_size[axis] * scale).is_integer():
                    raise ValueError(
                        f"For axis-{axis}, output[{ss}] will have non-integer shape. Spatial \
zoom_scale between output[{ss}] and input is {scale}. Please pad inputs."
                    )
                zoom_scale.append(scale)

            if not _initialized_list[ss]:  # init. buffer at the first iteration
                output_classes = seg_prob.shape[1]
                output_shape = [batch_size, output_classes] + [
                    int(round(image_size_d * zoom_scale_d))
                    for image_size_d, zoom_scale_d in zip(image_size, zoom_scale)
                ]
                # allocate memory to store the full output and the count for overlapping parts
                output_image_list.append(torch.zeros(output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                _initialized_list[ss] = True
                _initialized_list.append(False)

            # store the result in the proper location of the full output. Apply weights from importance map.
            for idx, original_idx in zip(slice_range, unravel_slice):
                # zoom roi
                original_idx_zoom = deepcopy(original_idx)  # 4D for 2D image, 5D for 3D image
                for axis in range(2, len(original_idx_zoom)):
                    zoomed_start = original_idx[axis].start * zoom_scale[axis - 2]
                    zoomed_end = original_idx[axis].stop * zoom_scale[axis - 2]
                    if not zoomed_start.is_integer() or (not zoomed_end.is_integer()):
                        raise ValueError(
                            f"For axis-{axis-2} of output[{ss}], the output roi range is not int. \
Input roi range is ({original_idx[axis].start}, {original_idx[axis].stop}). \
Spatial zoom_scale between output[{ss}] and input is {zoom_scale[axis - 2]}. \
Corresponding output roi range is ({zoomed_start}, {zoomed_end}).\n\
Please change overlap ({overlap}) or roi_size ({roi_size[axis-2]}) for axis-{axis-2}. \
Tips: if overlap*roi_size*zoom_scale is int, it usually works."
                        )
                    original_idx_zoom[axis] = slice(int(round(zoomed_start)), int(round(zoomed_end)), None)
                # zoom importance_map
                if importance_map.shape != seg_prob.shape[2:]:
                    importance_map_zoom = (
                        F.interpolate(
                            importance_map.unsqueeze(0).unsqueeze(0).to(torch.float32), size=seg_prob.shape[2:]
                        )
                        .to(compute_dtype)
                        .squeeze(0)
                        .squeeze(0)
                    )
                else:
                    importance_map_zoom = importance_map
                # store results and weights
                output_image_list[ss][original_idx_zoom] += importance_map_zoom * seg_prob[idx - slice_g]
                count_map_list[ss][original_idx_zoom] += (
                    importance_map_zoom.unsqueeze(0).unsqueeze(0).expand(count_map_list[ss][original_idx_zoom].shape)
                )

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] = (output_image_list[ss] / count_map_list[ss]).to(compute_dtype)
    del count_map_list

    for ss in range(len(output_image_list)):
        if torch.isnan(output_image_list[ss]).any() or torch.isinf(output_image_list[ss]).any():
            raise ValueError("Sliding window inference results contain NaN or Inf.")
        zoom_scale = [
            seg_prob_map_shape_d / roi_size_d
            for seg_prob_map_shape_d, roi_size_d in zip(output_image_list[ss].shape[2:], roi_size)
        ]
        final_slicing: List[slice] = []
        for sp in range(num_spatial_dims):
            slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
            slice_dim = slice(
                int(round(slice_dim.start * zoom_scale[num_spatial_dims - sp - 1])),
                int(round(slice_dim.stop * zoom_scale[num_spatial_dims - sp - 1])),
            )
            final_slicing.insert(0, slice_dim)
        while len(final_slicing) < len(output_image_list[ss].shape):
            final_slicing.insert(0, slice(None))
        output_image_list[ss] = output_image_list[ss][final_slicing]

    if dict_key is not None:
        final_output = {}
        for k, v in zip(dict_key, output_image_list):
            final_output[k] = v
    else:
        final_output = tuple(output_image_list)

    if tensor_output:
        return final_output[0]
    return final_output

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

import copy
from collections.abc import Sequence
from typing import Any

import torch

from monai.data.meta_tensor import MetaTensor
from monai.utils import optional_import

tqdm, _ = optional_import("tqdm", name="tqdm")

__all__ = ["point_based_window_inferer"]


def point_based_window_inferer(
    inputs: torch.Tensor | MetaTensor,
    roi_size: Sequence[int],
    predictor: torch.nn.Module,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    class_vector: torch.Tensor | None = None,
    prompt_class: torch.Tensor | None = None,
    prev_mask: torch.Tensor | MetaTensor | None = None,
    point_start: int = 0,
    center_only: bool = True,
    margin: int = 5,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Point-based window inferer that takes an input image, a set of points, and a model, and returns a segmented image.
    The inferer algorithm crops the input image into patches that centered at the point sets, which is followed by
    patch inference and average output stitching, and finally returns the segmented mask.

    Args:
        inputs: [1CHWD], input image to be processed.
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: the model. For vista3D, the output is [B, 1, H, W, D] which needs to be transposed to [1, B, H, W, D].
            Add transpose=True in kwargs for vista3d.
        point_coords: [B, N, 3]. Point coordinates for B foreground objects, each has N points.
        point_labels: [B, N]. Point labels. 0/1 means negative/positive points for regular supported or zero-shot classes.
            2/3 means negative/positive points for special supported classes (e.g. tumor, vessel).
        class_vector: [B]. Used for class-head automatic segmentation. Can be None value.
        prompt_class: [B]. The same as class_vector representing the point class and inform point head about
            supported class or zeroshot, not used for automatic segmentation. If None, point head is default
            to supported class segmentation.
        prev_mask: [1, B, H, W, D]. The value is before sigmoid. An optional tensor of previously segmented masks.
        point_start: only use points starting from this number. All points before this number is used to generate
            prev_mask. This is used to avoid re-calculating the points in previous iterations if given prev_mask.
        center_only: for each point, only crop the patch centered at this point. If false, crop 3 patches for each point.
        margin: if center_only is false, this value is the distance between point to the patch boundary.
    Returns:
        stitched_output: [1, B, H, W, D]. The value is before sigmoid.
    Notice: The function only supports SINGLE OBJECT INFERENCE with B=1.
    """
    if not point_coords.shape[0] == 1:
        raise ValueError("Only supports single object point click.")
    if not len(inputs.shape) == 5:
        raise ValueError("Input image should be 5D.")
    image, pad = _pad_previous_mask(copy.deepcopy(inputs), roi_size)
    point_coords = point_coords + torch.tensor([pad[-2], pad[-4], pad[-6]]).to(point_coords.device)
    prev_mask = _pad_previous_mask(copy.deepcopy(prev_mask), roi_size)[0] if prev_mask is not None else None
    stitched_output = None
    for p in point_coords[0][point_start:]:
        lx_, rx_ = _get_window_idx(p[0], roi_size[0], image.shape[-3], center_only=center_only, margin=margin)
        ly_, ry_ = _get_window_idx(p[1], roi_size[1], image.shape[-2], center_only=center_only, margin=margin)
        lz_, rz_ = _get_window_idx(p[2], roi_size[2], image.shape[-1], center_only=center_only, margin=margin)
        for i in range(len(lx_)):
            for j in range(len(ly_)):
                for k in range(len(lz_)):
                    lx, rx, ly, ry, lz, rz = (lx_[i], rx_[i], ly_[j], ry_[j], lz_[k], rz_[k])
                    unravel_slice = [
                        slice(None),
                        slice(None),
                        slice(int(lx), int(rx)),
                        slice(int(ly), int(ry)),
                        slice(int(lz), int(rz)),
                    ]
                    batch_image = image[unravel_slice]
                    output = predictor(
                        batch_image,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        class_vector=class_vector,
                        prompt_class=prompt_class,
                        patch_coords=[unravel_slice],
                        prev_mask=prev_mask,
                        **kwargs,
                    )
                    if stitched_output is None:
                        stitched_output = torch.zeros(
                            [1, output.shape[1], image.shape[-3], image.shape[-2], image.shape[-1]], device="cpu"
                        )
                        stitched_mask = torch.zeros(
                            [1, output.shape[1], image.shape[-3], image.shape[-2], image.shape[-1]], device="cpu"
                        )
                    stitched_output[unravel_slice] += output.to("cpu")
                    stitched_mask[unravel_slice] = 1
    # if stitched_mask is 0, then NaN value
    stitched_output = stitched_output / stitched_mask
    # revert padding
    stitched_output = stitched_output[
        :, :, pad[4] : image.shape[-3] - pad[5], pad[2] : image.shape[-2] - pad[3], pad[0] : image.shape[-1] - pad[1]
    ]
    stitched_mask = stitched_mask[
        :, :, pad[4] : image.shape[-3] - pad[5], pad[2] : image.shape[-2] - pad[3], pad[0] : image.shape[-1] - pad[1]
    ]
    if prev_mask is not None:
        prev_mask = prev_mask[
            :,
            :,
            pad[4] : image.shape[-3] - pad[5],
            pad[2] : image.shape[-2] - pad[3],
            pad[0] : image.shape[-1] - pad[1],
        ]
        prev_mask = prev_mask.to("cpu")  # type: ignore
        # for un-calculated place, use previous mask
        stitched_output[stitched_mask < 1] = prev_mask[stitched_mask < 1]
    if isinstance(inputs, torch.Tensor):
        inputs = MetaTensor(inputs)
    if not hasattr(stitched_output, "meta"):
        stitched_output = MetaTensor(stitched_output, affine=inputs.meta["affine"], meta=inputs.meta)
    return stitched_output


def _get_window_idx_c(p: int, roi: int, s: int) -> tuple[int, int]:
    """Helper function to get the window index."""
    if p - roi // 2 < 0:
        left, right = 0, roi
    elif p + roi // 2 > s:
        left, right = s - roi, s
    else:
        left, right = int(p) - roi // 2, int(p) + roi // 2
    return left, right


def _get_window_idx(p: int, roi: int, s: int, center_only: bool = True, margin: int = 5) -> tuple[list[int], list[int]]:
    """Get the window index."""
    left, right = _get_window_idx_c(p, roi, s)
    if center_only:
        return [left], [right]
    left_most = max(0, p - roi + margin)
    right_most = min(s, p + roi - margin)
    left_list = [left_most, right_most - roi, left]
    right_list = [left_most + roi, right_most, right]
    return left_list, right_list


def _pad_previous_mask(
    inputs: torch.Tensor | MetaTensor, roi_size: Sequence[int], padvalue: int = 0
) -> tuple[torch.Tensor | MetaTensor, list[int]]:
    """Helper function to pad inputs."""
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = torch.nn.functional.pad(inputs, pad=pad_size, mode="constant", value=padvalue)  # type: ignore
    return inputs, pad_size

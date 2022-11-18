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

# =========================================================================
# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/_utils.py
# which has the following license...
# https://github.com/pytorch/vision/blob/main/LICENSE
#
# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
This script is modified from torchvision to support N-D images,

https://github.com/pytorch/vision/blob/main/torchvision/models/detection/_utils.py
"""

import math
from typing import Sequence, Tuple, Union

import torch
from torch import Tensor

from monai.data.box_utils import COMPUTE_DTYPE, CenterSizeMode, StandardMode, convert_box_mode, is_valid_box_values
from monai.utils.module import look_up_option


def encode_boxes(gt_boxes: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some reference ground truth (gt) boxes.

    Args:
        gt_boxes: gt boxes, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``
        proposals: boxes to be encoded, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``
        weights: the weights for ``(cx, cy, w, h) or (cx,cy,cz, w,h,d)``

    Return:
        encoded gt, target of box regression that is used to convert proposals into gt_boxes, Nx4 or Nx6 torch tensor.
    """

    if gt_boxes.shape[0] != proposals.shape[0]:
        raise ValueError("gt_boxes.shape[0] should be equal to proposals.shape[0].")
    spatial_dims = look_up_option(len(weights), [4, 6]) // 2

    if not is_valid_box_values(gt_boxes):
        raise ValueError("gt_boxes is not valid. Please check if it contains empty boxes.")
    if not is_valid_box_values(proposals):
        raise ValueError("proposals is not valid. Please check if it contains empty boxes.")

    # implementation starts here
    ex_cccwhd: Tensor = convert_box_mode(proposals, src_mode=StandardMode, dst_mode=CenterSizeMode)  # type: ignore
    gt_cccwhd: Tensor = convert_box_mode(gt_boxes, src_mode=StandardMode, dst_mode=CenterSizeMode)  # type: ignore
    targets_dxyz = (
        weights[:spatial_dims].unsqueeze(0)
        * (gt_cccwhd[:, :spatial_dims] - ex_cccwhd[:, :spatial_dims])
        / ex_cccwhd[:, spatial_dims:]
    )
    targets_dwhd = weights[spatial_dims:].unsqueeze(0) * torch.log(
        gt_cccwhd[:, spatial_dims:] / ex_cccwhd[:, spatial_dims:]
    )

    targets = torch.cat((targets_dxyz, targets_dwhd), dim=1)
    # torch.log may cause NaN or Inf
    if torch.isnan(targets).any() or torch.isinf(targets).any():
        raise ValueError("targets is NaN or Inf.")
    return targets


class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.

    Args:
        weights: 4-element tuple or 6-element tuple
        boxes_xform_clip: high threshold to prevent sending too large values into torch.exp()

    Example:
        .. code-block:: python

            box_coder = BoxCoder(weights=[1., 1., 1., 1., 1., 1.])
            gt_boxes = torch.tensor([[1,2,1,4,5,6],[1,3,2,7,8,9]])
            proposals = gt_boxes + torch.rand(gt_boxes.shape)
            rel_gt_boxes = box_coder.encode_single(gt_boxes, proposals)
            gt_back = box_coder.decode_single(rel_gt_boxes, proposals)
            # We expect gt_back to be equal to gt_boxes
    """

    def __init__(self, weights: Tuple[float], boxes_xform_clip: Union[float, None] = None) -> None:
        if boxes_xform_clip is None:
            boxes_xform_clip = math.log(1000.0 / 16)
        self.spatial_dims = look_up_option(len(weights), [4, 6]) // 2
        self.weights = weights
        self.boxes_xform_clip = boxes_xform_clip

    def encode(self, gt_boxes: Sequence[Tensor], proposals: Sequence[Tensor]) -> Tuple[Tensor]:
        """
        Encode a set of proposals with respect to some ground truth (gt) boxes.

        Args:
            gt_boxes: list of gt boxes, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``
            proposals: list of boxes to be encoded, each element is Mx4 or Mx6 torch tensor.
                The box mode is assumed to be ``StandardMode``

        Return:
            A tuple of encoded gt, target of box regression that is used to
                convert proposals into gt_boxes, Nx4 or Nx6 torch tensor.
        """
        boxes_per_image = [len(b) for b in gt_boxes]
        # concat the lists to do computation
        concat_gt_boxes = torch.cat(tuple(gt_boxes), dim=0)
        concat_proposals = torch.cat(tuple(proposals), dim=0)
        concat_targets = self.encode_single(concat_gt_boxes, concat_proposals)
        # split to tuple
        targets: Tuple[Tensor] = concat_targets.split(boxes_per_image, 0)
        return targets

    def encode_single(self, gt_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode proposals with respect to ground truth (gt) boxes.

        Args:
            gt_boxes: gt boxes, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``
            proposals: boxes to be encoded, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``

        Return:
            encoded gt, target of box regression that is used to convert proposals into gt_boxes, Nx4 or Nx6 torch tensor.
        """
        dtype = gt_boxes.dtype
        device = gt_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(gt_boxes, proposals, weights)
        return targets

    def decode(self, rel_codes: Tensor, reference_boxes: Sequence[Tensor]) -> Tensor:
        """
        From a set of original reference_boxes and encoded relative box offsets,

        Args:
            rel_codes: encoded boxes, Nx4 or Nx6 torch tensor.
            reference_boxes: a list of reference boxes, each element is Mx4 or Mx6 torch tensor.
                The box mode is assumed to be ``StandardMode``

        Return:
            decoded boxes, Nx1x4 or Nx1x6 torch tensor. The box mode will be ``StandardMode``
        """
        if not isinstance(reference_boxes, Sequence) or (not isinstance(rel_codes, torch.Tensor)):
            raise ValueError("Input arguments wrong type.")
        boxes_per_image = [b.size(0) for b in reference_boxes]
        # concat the lists to do computation
        concat_boxes = torch.cat(tuple(reference_boxes), dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 2 * self.spatial_dims)
        return pred_boxes

    def decode_single(self, rel_codes: Tensor, reference_boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,

        Args:
            rel_codes: encoded boxes, Nx(4*num_box_reg) or Nx(6*num_box_reg) torch tensor.
            reference_boxes: reference boxes, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``

        Return:
            decoded boxes, Nx(4*num_box_reg) or Nx(6*num_box_reg) torch tensor. The box mode will to be ``StandardMode``
        """
        reference_boxes = reference_boxes.to(rel_codes.dtype)
        offset = reference_boxes.shape[-1]

        pred_boxes = []
        boxes_cccwhd = convert_box_mode(reference_boxes, src_mode=StandardMode, dst_mode=CenterSizeMode)
        for axis in range(self.spatial_dims):
            whd_axis = boxes_cccwhd[:, axis + self.spatial_dims]
            ctr_xyz_axis = boxes_cccwhd[:, axis]
            dxyz_axis = rel_codes[:, axis::offset] / self.weights[axis]
            dwhd_axis = rel_codes[:, self.spatial_dims + axis :: offset] / self.weights[axis + self.spatial_dims]
            # Prevent sending too large values into torch.exp()
            dwhd_axis = torch.clamp(dwhd_axis.to(COMPUTE_DTYPE), max=self.boxes_xform_clip)

            pred_ctr_xyx_axis = dxyz_axis * whd_axis[:, None] + ctr_xyz_axis[:, None]
            pred_whd_axis = torch.exp(dwhd_axis) * whd_axis[:, None]
            pred_whd_axis = pred_whd_axis.to(dxyz_axis.dtype)

            # When convert float32 to float16, Inf or Nan may occur
            if torch.isnan(pred_whd_axis).any() or torch.isinf(pred_whd_axis).any():
                raise ValueError("pred_whd_axis is NaN or Inf.")

            # Distance from center to box's corner.
            c_to_c_whd_axis = (
                torch.tensor(0.5, dtype=pred_ctr_xyx_axis.dtype, device=pred_whd_axis.device) * pred_whd_axis
            )

            pred_boxes.append(pred_ctr_xyx_axis - c_to_c_whd_axis)
            pred_boxes.append(pred_ctr_xyx_axis + c_to_c_whd_axis)

        pred_boxes = pred_boxes[::2] + pred_boxes[1::2]
        pred_boxes_final = torch.stack(pred_boxes, dim=2).flatten(1)
        return pred_boxes_final

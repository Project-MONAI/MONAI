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
# Adapted from https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/matcher.py
# which has the following license...
# https://github.com/MIC-DKFZ/nnDetection/blob/main/LICENSE
#
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
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
The functions in this script are adapted from nnDetection,
https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/matcher.py
which is adapted from torchvision.

These are the changes compared with nndetection:
1) comments and docstrings;
2) reformat;
3) add a debug option to ATSSMatcher to help the users to tune parameters;
4) add a corner case return in ATSSMatcher.compute_matches;
5) add support for float16 cpu
"""

import logging
from abc import ABC
from typing import Callable, Sequence, Tuple, TypeVar

import torch
from torch import Tensor

from monai.data.box_utils import COMPUTE_DTYPE, box_iou, boxes_center_distance, centers_in_boxes
from monai.utils.type_conversion import convert_to_tensor

# -INF should be smaller than the lower bound of similarity_fn output.
INF = float("inf")


class Matcher(ABC):
    """
    Base class of Matcher, which matches boxes and anchors to each other

    Args:
        similarity_fn: function for similarity computation between
            boxes and anchors
    """
    BELOW_LOW_THRESHOLD: int = -1
    BETWEEN_THRESHOLDS: int = -2

    def __init__(self, similarity_fn: Callable[[Tensor, Tensor], Tensor] = box_iou):  # type: ignore
        self.similarity_fn = similarity_fn

    def __call__(
        self, boxes: torch.Tensor, anchors: torch.Tensor, num_anchors_per_level: Sequence[int], num_anchors_per_loc: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches for a single image

        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            anchors: anchors to match Mx4 or Mx6, also assumed to be ``StandardMode``.
            num_anchors_per_level: number of anchors per feature pyramid level
            num_anchors_per_loc: number of anchors per position

        Returns:
            - matrix which contains the similarity from each boxes
                to each anchor [N, M]
            - vector which contains the matched box index for all
                anchors (if background `BELOW_LOW_THRESHOLD` is used
                and if it should be ignored `BETWEEN_THRESHOLDS` is used)
                [M]

        Note:
            ``StandardMode`` = :class:`~monai.data.box_utils.CornerCornerModeTypeA`,
            also represented as "xyxy" ([xmin, ymin, xmax, ymax]) for 2D
            and "xyzxyz" ([xmin, ymin, zmin, xmax, ymax, zmax]) for 3D.
        """
        if boxes.numel() == 0:
            # no ground truth
            num_anchors = anchors.shape[0]
            match_quality_matrix = torch.tensor([]).to(anchors)
            matches = torch.empty(num_anchors, dtype=torch.int64).fill_(self.BELOW_LOW_THRESHOLD)
            return match_quality_matrix, matches
        else:
            # at least one ground truth
            return self.compute_matches(
                boxes=boxes,
                anchors=anchors,
                num_anchors_per_level=num_anchors_per_level,
                num_anchors_per_loc=num_anchors_per_loc,
            )

    def compute_matches(
        self, boxes: torch.Tensor, anchors: torch.Tensor, num_anchors_per_level: Sequence[int], num_anchors_per_loc: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches

        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            anchors: anchors to match Mx4 or Mx6, also assumed to be ``StandardMode``.
            num_anchors_per_level: number of anchors per feature pyramid level
            num_anchors_per_loc: number of anchors per position

        Returns:
            - matrix which contains the similarity from each boxes
                to each anchor [N, M]
            - vector which contains the matched box index for all
                anchors (if background `BELOW_LOW_THRESHOLD` is used
                and if it should be ignored `BETWEEN_THRESHOLDS` is used)
                [M]
        """
        raise NotImplementedError


class ATSSMatcher(Matcher):
    def __init__(
        self,
        num_candidates: int = 4,
        similarity_fn: Callable[[Tensor, Tensor], Tensor] = box_iou,  # type: ignore
        center_in_gt: bool = True,
        debug: bool = False,
    ):
        """
        Compute matching based on ATSS
        https://arxiv.org/abs/1912.02424
        `Bridging the Gap Between Anchor-based and Anchor-free Detection
        via Adaptive Training Sample Selection`

        Args:
            num_candidates: number of positions to select candidates from.
                Smaller value will result in a higher matcher threshold and less matched candidates.
            similarity_fn: function for similarity computation between
                boxes and anchors
            center_in_gt: If False (default), matched anchor center points do not need
                to lie withing the ground truth box. Recommand False for small objects.
                If True, will result in a strict matcher and less matched candidates.
            debug: if True, will print the matcher threshold in order to
                tune ``num_candidates`` and ``center_in_gt``.
        """
        super().__init__(similarity_fn=similarity_fn)
        self.num_candidates = num_candidates
        self.min_dist = 0.01
        self.center_in_gt = center_in_gt
        self.debug = debug
        logging.info(
            f"Running ATSS Matching with num_candidates={self.num_candidates} " f"and center_in_gt {self.center_in_gt}."
        )

    def compute_matches(
        self, boxes: torch.Tensor, anchors: torch.Tensor, num_anchors_per_level: Sequence[int], num_anchors_per_loc: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches according to ATTS for a single image
        Adapted from
        (https://github.com/sfzhang15/ATSS/blob/79dfb28bd1/atss_core/modeling/rpn/atss
        /loss.py#L180-L184)

        Args:
            boxes: bounding boxes, Nx4 or Nx6 torch tensor or ndarray. The box mode is assumed to be ``StandardMode``
            anchors: anchors to match Mx4 or Mx6, also assumed to be ``StandardMode``.
            num_anchors_per_level: number of anchors per feature pyramid level
            num_anchors_per_loc: number of anchors per position

        Returns:
            Tensor: matrix which contains the similarity from each boxes
                to each anchor [N, M]
            Tensor: vector which contains the matched box index for all
                anchors (if background `BELOW_LOW_THRESHOLD` is used
                and if it should be ignored `BETWEEN_THRESHOLDS` is used)
                [M]

        Note:
            ``StandardMode`` = :class:`~monai.data.box_utils.CornerCornerModeTypeA`,
            also represented as "xyxy" ([xmin, ymin, xmax, ymax]) for 2D
            and "xyzxyz" ([xmin, ymin, zmin, xmax, ymax, zmax]) for 3D.
        """
        num_gt = boxes.shape[0]
        num_anchors = anchors.shape[0]

        distances_, _, anchors_center = boxes_center_distance(boxes, anchors)  # num_boxes x anchors
        distances = convert_to_tensor(distances_)

        # select candidates based on center distance
        candidate_idx_list = []
        start_idx = 0
        for _, apl in enumerate(num_anchors_per_level):
            end_idx = start_idx + apl * num_anchors_per_loc

            # topk: total number of candidates per position
            topk = min(self.num_candidates * num_anchors_per_loc, apl)
            # torch.topk() does not support float16 cpu, need conversion to float32 or float64
            _, idx = distances[:, start_idx:end_idx].to(COMPUTE_DTYPE).topk(topk, dim=1, largest=False)
            # idx: shape [num_boxes x topk]
            candidate_idx_list.append(idx + start_idx)

            start_idx = end_idx
        # [num_boxes x num_candidates] (index of candidate anchors)
        candidate_idx = torch.cat(candidate_idx_list, dim=1)

        match_quality_matrix = self.similarity_fn(boxes, anchors)  # [num_boxes x anchors]
        candidate_ious = match_quality_matrix.gather(1, candidate_idx)  # [num_boxes, n_candidates]

        # corner case, n_candidates<=1 will make iou_std_per_gt NaN
        if candidate_idx.shape[1] <= 1:
            matches = -1 * torch.ones((num_anchors,), dtype=torch.long, device=boxes.device)
            matches[candidate_idx] = 0
            return match_quality_matrix, matches

        # compute adaptive iou threshold
        iou_mean_per_gt = candidate_ious.mean(dim=1)  # [num_boxes]
        iou_std_per_gt = candidate_ious.std(dim=1)  # [num_boxes]
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt  # [num_boxes]
        is_pos = candidate_ious >= iou_thresh_per_gt[:, None]  # [num_boxes x n_candidates]
        if self.debug:
            print(f"Anchor matcher threshold: {iou_thresh_per_gt}")

        if self.center_in_gt:  # can discard all candidates in case of very small objects :/
            # center point of selected anchors needs to lie within the ground truth
            boxes_idx = (
                torch.arange(num_gt, device=boxes.device, dtype=torch.long)[:, None]
                .expand_as(candidate_idx)
                .contiguous()
            )  # [num_boxes x n_candidates]
            is_in_gt_ = centers_in_boxes(
                anchors_center[candidate_idx.view(-1)], boxes[boxes_idx.view(-1)], eps=self.min_dist
            )
            is_in_gt = convert_to_tensor(is_in_gt_)
            is_pos = is_pos & is_in_gt.view_as(is_pos)  # [num_boxes x n_candidates]

        # in case on anchor is assigned to multiple boxes, use box with highest IoU
        # TODO: think about a better way to do this
        for ng in range(num_gt):
            candidate_idx[ng, :] += ng * num_anchors
        ious_inf = torch.full_like(match_quality_matrix, -INF).view(-1)
        index = candidate_idx.view(-1)[is_pos.view(-1)]
        ious_inf[index] = match_quality_matrix.view(-1)[index]
        ious_inf = ious_inf.view_as(match_quality_matrix)

        matched_vals, matches = ious_inf.to(COMPUTE_DTYPE).max(dim=0)
        matches[matched_vals == -INF] = self.BELOW_LOW_THRESHOLD
        # print(f"Num matches {(matches >= 0).sum()}, Adapt IoU {iou_thresh_per_gt}")
        return match_quality_matrix, matches


MatcherType = TypeVar("MatcherType", bound=Matcher)

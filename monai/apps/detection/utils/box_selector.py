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
# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
# which has the following license...
# https://github.com/pytorch/vision/blob/main/LICENSE

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

"""
Part of this script is adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
"""

from typing import Callable, List, Tuple, Union

import torch
from torch import Tensor

from monai.data.box_utils import batched_nms, box_iou, clip_boxes_to_image


class BoxSelector:
    """
    Box selector which selects the predicted boxes.
    The box selection is performed with the following steps:

    #. For each level, discard boxes with scores less than self.score_thresh.
    #. For each level, keep boxes with top self.topk_candidates_per_level scores.
    #. For the whole image, perform non-maximum suppression (NMS) on boxes, with overapping threshold nms_thresh.
    #. For the whole image, keep boxes with top self.detections_per_img scores.

    Args:
        apply_sigmoid: whether to apply sigmoid to get scores from classification logits
        score_thresh: no box with scores less than score_thresh will be kept
        topk_candidates_per_level: max number of boxes to keep for each level
        nms_thresh: box overlapping threshold for NMS
        detections_per_img: max number of boxes to keep for each image

    Example:

        .. code-block:: python

            input_param = {
                "apply_sigmoid": True,
                "score_thresh": 0.1,
                "topk_candidates_per_level": 2,
                "nms_thresh": 0.1,
                "detections_per_img": 5,
            }
            box_selector = BoxSelector(**input_param)
            boxes = [torch.randn([3,6]), torch.randn([7,6])]
            logits = [torch.randn([3,3]), torch.randn([7,3])]
            spatial_size = (8,8,8)
            selected_boxes, selected_scores, selected_labels = box_selector.select_boxes_per_image(
                boxes, logits, spatial_size
            )
    """

    def __init__(
        self,
        box_overlap_metric: Callable = box_iou,
        apply_sigmoid: bool = True,
        score_thresh=0.05,
        topk_candidates_per_level=1000,
        nms_thresh=0.5,
        detections_per_img=300,
    ):
        self.box_overlap_metric = box_overlap_metric

        self.apply_sigmoid = apply_sigmoid
        self.score_thresh = score_thresh
        self.topk_candidates_per_level = topk_candidates_per_level
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def select_top_score_idx_per_level(self, logits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Select indice with highest scores.

        The indice selection is performed with the following steps:

        #. If self.apply_sigmoid, get scores by applying sigmoid to logits. Otherwise, use logits as scores.
        #. Discard indice with scores less than self.score_thresh
        #. Keep indice with top self.topk_candidates_per_level scores

        Args:
            logits: predicted classification logits, Tensor sized (N, num_classes)

        Return:
            - topk_idxs: selected M indices, Tensor sized (M, )
            - selected_scores: selected M scores, Tensor sized (M, )
            - selected_labels: selected M labels, Tensor sized (M, )
        """
        num_classes = logits.shape[-1]

        # apply sigmoid to classification logits if asked
        if self.apply_sigmoid:
            scores = torch.sigmoid(logits.to(torch.float32)).flatten()
        else:
            scores = logits.flatten()

        # remove low scoring boxes
        keep_idxs = scores > self.score_thresh
        scores = scores[keep_idxs]
        flatten_topk_idxs = torch.where(keep_idxs)[0]

        # keep only topk scoring predictions
        num_topk = min(self.topk_candidates_per_level, flatten_topk_idxs.size(0))
        selected_scores, idxs = scores.to(torch.float32).topk(
            num_topk
        )  # half precision not implemented for cpu float16
        flatten_topk_idxs = flatten_topk_idxs[idxs]

        selected_labels = flatten_topk_idxs % num_classes

        topk_idxs = torch.div(flatten_topk_idxs, num_classes, rounding_mode="floor")
        return topk_idxs, selected_scores, selected_labels

    def select_boxes_per_image(
        self, boxes_list: List[Tensor], logits_list: List[Tensor], spatial_size: Union[List[int], Tuple[int]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Postprocessing to generate detection result from classification logits and boxes.

        The box selection is performed with the following steps:

        #. For each level, discard boxes with scores less than self.score_thresh.
        #. For each level, keep boxes with top self.topk_candidates_per_level scores.
        #. For the whole image, perform non-maximum suppression (NMS) on boxes, with overapping threshold nms_thresh.
        #. For the whole image, keep boxes with top self.detections_per_img scores.

        Args:
            boxes_list: list of predicted boxes from a single image,
                each element i is a Tensor sized (N_i, 2*spatial_dims)
            logits_list: list of predicted classification logits from a single image,
                each element i is a Tensor sized (N_i, num_classes)
            spatial_size: spatial size of the image

        Return:
            - selected boxes, Tensor sized (P, 2*spatial_dims)
            - selected_scores, Tensor sized (P, )
            - selected_labels, Tensor sized (P, )
        """

        if len(boxes_list) != len(logits_list):
            raise ValueError(
                "len(boxes_list) should equal to len(logits_list). "
                f"Got len(boxes_list)={len(boxes_list)}, len(logits_list)={len(logits_list)}"
            )

        image_boxes = []
        image_scores = []
        image_labels = []

        compute_dtype = boxes_list[0].dtype

        for boxes_per_level, logits_per_level in zip(boxes_list, logits_list):
            # select topk boxes for each level
            topk_idxs: Tensor
            topk_idxs, scores_per_level, labels_per_level = self.select_top_score_idx_per_level(logits_per_level)
            boxes_per_level = boxes_per_level[topk_idxs]

            keep: Tensor
            boxes_per_level, keep = clip_boxes_to_image(boxes_per_level, spatial_size, remove_empty=True)  # type: ignore
            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level[keep])
            image_labels.append(labels_per_level[keep])

        image_boxes_t: Tensor = torch.cat(image_boxes, dim=0)
        image_scores_t: Tensor = torch.cat(image_scores, dim=0)
        image_labels_t: Tensor = torch.cat(image_labels, dim=0)

        # non-maximum suppression on detected boxes from all levels
        keep_t: Tensor = batched_nms(  # type: ignore
            image_boxes_t,
            image_scores_t,
            image_labels_t,
            self.nms_thresh,
            box_overlap_metric=self.box_overlap_metric,
            max_proposals=self.detections_per_img,
        )

        selected_boxes = image_boxes_t[keep_t].to(compute_dtype)
        selected_scores = image_scores_t[keep_t].to(compute_dtype)
        selected_labels = image_labels_t[keep_t]

        return selected_boxes, selected_scores, selected_labels

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

from typing import Any, cast

import numpy as np
import torch

from monai.config import NdarrayOrTensor


def compute_fp_tp_probs_nd(
    probs: NdarrayOrTensor,
    coords: NdarrayOrTensor,
    evaluation_mask: NdarrayOrTensor,
    labels_to_exclude: list | None = None,
) -> tuple[NdarrayOrTensor, NdarrayOrTensor, int]:
    """
    This function is modified from the official evaluation code of
    `CAMELYON 16 Challenge <https://camelyon16.grand-challenge.org/>`_, and used to distinguish
    true positive and false positive predictions. A true positive prediction is defined when
    the detection point is within the annotated ground truth region.

    Args:
        probs: an array with shape (n,) that represents the probabilities of the detections.
            Where, n is the number of predicted detections.
        coords: an array with shape (n, n_dim) that represents the coordinates of the detections.
            The dimensions must be in the same order as in `evaluation_mask`.
        evaluation_mask: the ground truth mask for evaluation.
        labels_to_exclude: labels in this list will not be counted for metric calculation.

    Returns:
        fp_probs: an array that contains the probabilities of the false positive detections.
        tp_probs: an array that contains the probabilities of the True positive detections.
        num_targets: the total number of targets (excluding `labels_to_exclude`) for all images under evaluation.

    """
    if not (len(probs) == len(coords)):
        raise ValueError(f"the length of probs {probs.shape}, should be the same as of coords {coords.shape}.")
    if not (len(coords.shape) > 1 and coords.shape[1] == len(evaluation_mask.shape)):
        raise ValueError(
            f"coords {coords.shape} need to represent the same number of dimensions as mask {evaluation_mask.shape}."
        )

    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    if isinstance(evaluation_mask, torch.Tensor):
        evaluation_mask = evaluation_mask.detach().cpu().numpy()

    if labels_to_exclude is None:
        labels_to_exclude = []

    max_label = np.max(evaluation_mask)
    tp_probs = np.zeros((max_label,), dtype=np.float32)

    hittedlabel = evaluation_mask[tuple(coords.T)]
    fp_probs = probs[np.where(hittedlabel == 0)]
    for i in range(1, max_label + 1):
        if i not in labels_to_exclude and i in hittedlabel:
            tp_probs[i - 1] = probs[np.where(hittedlabel == i)].max()

    num_targets = max_label - len(labels_to_exclude)
    return fp_probs, tp_probs, cast(int, num_targets)


def compute_fp_tp_probs(
    probs: NdarrayOrTensor,
    y_coord: NdarrayOrTensor,
    x_coord: NdarrayOrTensor,
    evaluation_mask: NdarrayOrTensor,
    labels_to_exclude: list | None = None,
    resolution_level: int = 0,
) -> tuple[NdarrayOrTensor, NdarrayOrTensor, int]:
    """
    This function is modified from the official evaluation code of
    `CAMELYON 16 Challenge <https://camelyon16.grand-challenge.org/>`_, and used to distinguish
    true positive and false positive predictions. A true positive prediction is defined when
    the detection point is within the annotated ground truth region.

    Args:
        probs: an array with shape (n,) that represents the probabilities of the detections.
            Where, n is the number of predicted detections.
        y_coord: an array with shape (n,) that represents the Y-coordinates of the detections.
        x_coord: an array with shape (n,) that represents the X-coordinates of the detections.
        evaluation_mask: the ground truth mask for evaluation.
        labels_to_exclude: labels in this list will not be counted for metric calculation.
        resolution_level: the level at which the evaluation mask is made.

    Returns:
        fp_probs: an array that contains the probabilities of the false positive detections.
        tp_probs: an array that contains the probabilities of the True positive detections.
        num_targets: the total number of targets (excluding `labels_to_exclude`) for all images under evaluation.

    """
    if isinstance(y_coord, torch.Tensor):
        y_coord = y_coord.detach().cpu().numpy()
    if isinstance(x_coord, torch.Tensor):
        x_coord = x_coord.detach().cpu().numpy()

    y_coord = (y_coord / pow(2, resolution_level)).astype(int)
    x_coord = (x_coord / pow(2, resolution_level)).astype(int)

    stacked = np.stack([y_coord, x_coord], axis=1)

    return compute_fp_tp_probs_nd(
        probs=probs, coords=stacked, evaluation_mask=evaluation_mask, labels_to_exclude=labels_to_exclude
    )


def compute_froc_curve_data(
    fp_probs: np.ndarray | torch.Tensor, tp_probs: np.ndarray | torch.Tensor, num_targets: int, num_images: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function is modified from the official evaluation code of
    `CAMELYON 16 Challenge <https://camelyon16.grand-challenge.org/>`_, and used to compute
    the required data for plotting the Free Response Operating Characteristic (FROC) curve.

    Args:
        fp_probs: an array that contains the probabilities of the false positive detections for all
            images under evaluation.
        tp_probs: an array that contains the probabilities of the True positive detections for all
            images under evaluation.
        num_targets: the total number of targets (excluding `labels_to_exclude`) for all images under evaluation.
        num_images: the number of images under evaluation.

    """
    if not isinstance(fp_probs, type(tp_probs)):
        raise AssertionError("fp and tp probs should have same type.")
    if isinstance(fp_probs, torch.Tensor):
        fp_probs = fp_probs.detach().cpu().numpy()
    if isinstance(tp_probs, torch.Tensor):
        tp_probs = tp_probs.detach().cpu().numpy()

    total_fps, total_tps = [], []
    all_probs = sorted(set(list(fp_probs) + list(tp_probs)))
    for thresh in all_probs[1:]:
        total_fps.append((fp_probs >= thresh).sum())
        total_tps.append((tp_probs >= thresh).sum())
    total_fps.append(0)
    total_tps.append(0)
    fps_per_image = np.asarray(total_fps) / float(num_images)
    total_sensitivity = np.asarray(total_tps) / float(num_targets)
    return fps_per_image, total_sensitivity


def compute_froc_score(
    fps_per_image: np.ndarray, total_sensitivity: np.ndarray, eval_thresholds: tuple = (0.25, 0.5, 1, 2, 4, 8)
) -> Any:
    """
    This function is modified from the official evaluation code of
    `CAMELYON 16 Challenge <https://camelyon16.grand-challenge.org/>`_, and used to compute
    the challenge's second evaluation metric, which is defined as the average sensitivity at
    the predefined false positive rates per whole slide image.

    Args:
        fps_per_image: the average number of false positives per image for different thresholds.
        total_sensitivity: sensitivities (true positive rates) for different thresholds.
        eval_thresholds: the false positive rates for calculating the average sensitivity. Defaults
            to (0.25, 0.5, 1, 2, 4, 8) which is the same as the CAMELYON 16 Challenge.

    """
    interp_sens = np.interp(eval_thresholds, fps_per_image[::-1], total_sensitivity[::-1])
    return np.mean(interp_sens)

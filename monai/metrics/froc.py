# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import numpy as np
import torch


def compute_fp_tp_probs(
    probs: Union[np.ndarray, torch.Tensor],
    ycorr: Union[np.ndarray, torch.Tensor],
    xcorr: Union[np.ndarray, torch.Tensor],
    evaluation_mask: Union[np.ndarray, torch.Tensor],
    isolated_tumor_cells: Optional[List] = None,
    image_level: int = 0,
):
    """
    This function is modified from the official evaluation code of
    `CAMELYON 16 Challenge <https://camelyon16.grand-challenge.org/>`_, and used to distinguish
    true positive and false positive predictions. A true positive prediction is defined when
    the detection point is within the annotated ground truth region.

    Args:
        probs: an array with shape (n,) that represents the probabilities of the detections.
            Where, n is the number of predicted detections.
        ycorr: an array with shape (n,) that represents the Y-coordinates of the detections.
        xcorr: an array with shape (n,) that represents the X-coordinates of the detections.
        evaluation_mask: the ground truth mask for evaluation.
        isolated_tumor_cells: a list of labels (of the evaluation mask)
            that contains Isolated Tumor Cells.
        image_level: the level at which the evaluation mask is made.

    Returns:
        fp_probs: an array that contains the probabilities of the false positive detections.
        tp_probs: an array that contains the probabilities of the True positive detections.
        num_of_tumors: number of tumors in the image (excluding Isolate Tumor Cells).

    """
    assert probs.shape == ycorr.shape == xcorr.shape, "the shapes for coordinates and probabilities should be the same."

    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    if torch.is_tensor(ycorr):
        ycorr = ycorr.detach().cpu().numpy()
    if torch.is_tensor(xcorr):
        xcorr = xcorr.detach().cpu().numpy()
    if torch.is_tensor(evaluation_mask):
        evaluation_mask = evaluation_mask.detach().cpu().numpy()

    if isolated_tumor_cells is None:
        isolated_tumor_cells = []

    max_label = np.max(evaluation_mask)
    tp_probs = np.zeros((max_label,), dtype=np.float32)

    ycorr = (ycorr / pow(2, image_level)).astype(int)
    xcorr = (xcorr / pow(2, image_level)).astype(int)

    hittedlabel = evaluation_mask[ycorr, xcorr]
    fp_probs = probs[np.where(hittedlabel == 0)]
    for i in range(1, max_label + 1):
        if i not in isolated_tumor_cells and i in hittedlabel:
            tp_probs[i - 1] = probs[np.where(hittedlabel == i)].max()

    num_of_tumors = max_label - len(isolated_tumor_cells)
    return fp_probs, tp_probs, num_of_tumors


def compute_froc_curve_data(
    fp_probs: Union[np.ndarray, torch.Tensor],
    tp_probs: Union[np.ndarray, torch.Tensor],
    num_of_tumors: int,
    num_of_samples: int,
):
    """
    This function is modified from the official evaluation code of
    `CAMELYON 16 Challenge <https://camelyon16.grand-challenge.org/>`_, and used to compute
    the required data for plotting the Free Response Operating Characteristic (FROC) curve.

    Args:
        fp_probs: an array that contains the probabilities of the false positive detections for all
            samples that to be computed.
        tp_probs: an array that contains the probabilities of the True positive detections for all
            samples that to be computed.
        num_of_tumors: the total number of tumors (excluding Isolate Tumor Cells) for all samples
            that to be computed.
        num_of_samples: the number of samples that to be computed.

    """
    assert type(fp_probs) == type(tp_probs), "fp and tp probs should have same type."
    if torch.is_tensor(fp_probs):
        fp_probs = fp_probs.detach().cpu().numpy()
    if torch.is_tensor(tp_probs):
        tp_probs = tp_probs.detach().cpu().numpy()

    total_fps, total_tps = [], []
    all_probs = sorted(set(list(fp_probs) + list(tp_probs)))
    for thresh in all_probs[1:]:
        total_fps.append((fp_probs >= thresh).sum())
        total_tps.append((tp_probs >= thresh).sum())
    total_fps.append(0)
    total_tps.append(0)
    fps_per_image = np.asarray(total_fps) / float(num_of_samples)
    total_sensitivity = np.asarray(total_tps) / float(num_of_tumors)
    return fps_per_image, total_sensitivity


def compute_froc_score(
    fps_per_image: np.ndarray,
    total_sensitivity: np.ndarray,
    eval_thresholds: Tuple = (0.25, 0.5, 1, 2, 4, 8),
):
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

"""
Some parts are adapted from https://github.com/cocodataset/cocoapi :

Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies, 
either expressed or implied, of the FreeBSD Project.
"""
"""
For the remaining parts:

Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from typing import Callable, Sequence, List, Dict


__all__ = ["matching_batch"]


def matching_batch(
    iou_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], 
    iou_thresholds: Sequence[float], pred_boxes: Sequence[np.ndarray],
    pred_classes: Sequence[np.ndarray], pred_scores: Sequence[np.ndarray],
    gt_boxes: Sequence[np.ndarray], gt_classes: Sequence[np.ndarray],
    gt_ignore: Sequence[Sequence[bool]], max_detections: int = 100,
    ) -> List[Dict[int, Dict[str, np.ndarray]]]:
    """
    Match boxes of a batch to corresponding ground truth for each category
    independently

    Args:
        iou_fn: compute overlap for each pair
        iou_thresholds: defined which IoU thresholds should be evaluated
        pred_boxes: predicted boxes from single batch; List[[D, dim * 2]],
            D number of predictions
        pred_classes: predicted classes from a single batch; List[[D]],
            D number of predictions
        pred_scores: predicted score for each bounding box; List[[D]],
            D number of predictions
        gt_boxes: ground truth boxes; List[[G, dim * 2]], G number of ground
            truth
        gt_classes: ground truth classes; List[[G]], G number of ground truth
        gt_ignore: specified if which ground truth boxes are not counted as
            true positives
            (detections which match theses boxes are not counted as false
            positives either); List[[G]], G number of ground truth
        max_detections: maximum number of detections which should be evaluated

    Returns:
        List[Dict[int, Dict[str, np.ndarray]]]
            matched detections [dtMatches] and ground truth [gtMatches]
            boxes [str, np.ndarray] for each category (stored in dict keys)
            for each image (list)
    """
    results = []
    # iterate over images/batches
    for pboxes, pclasses, pscores, gboxes, gclasses, gignore in zip(
        pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, gt_ignore):
        img_classes = np.union1d(pclasses, gclasses)
        result = {}  # dict contains results for each class in one image
        for c in img_classes:
            pred_mask = pclasses == c # mask predictions with current class
            gt_mask = gclasses == c # mask ground trtuh with current class

            if not np.any(gt_mask): # no ground truth
                result[c] = _matching_no_gt(
                    iou_thresholds=iou_thresholds,
                    pred_scores=pscores[pred_mask],
                    max_detections=max_detections)
            elif not np.any(pred_mask): # no predictions
                result[c] = _matching_no_pred(
                    iou_thresholds=iou_thresholds,
                    gt_ignore=gignore[gt_mask],
                )
            else: # at least one prediction and one ground truth
                result[c] = _matching_single_image_single_class(
                    iou_fn=iou_fn,
                    pred_boxes=pboxes[pred_mask],
                    pred_scores=pscores[pred_mask],
                    gt_boxes=gboxes[gt_mask],
                    gt_ignore=gignore[gt_mask],
                    max_detections=max_detections,
                    iou_thresholds=iou_thresholds,
                )
        results.append(result)
    return results


def _matching_no_gt(
        iou_thresholds: Sequence[float],
        pred_scores: np.ndarray,
        max_detections: int,
        ):
    """
    Matching result with not ground truth in image

    Args:
        iou_thresholds: defined which IoU thresholds should be evaluated
        dt_scores: predicted scores
        max_detections: maximum number of allowed detections per image.
            This functions uses this parameter to stay consistent with
            the actual matching function which needs this limit.

    Returns:
        dict: computed matching
            `dtMatches`: matched detections [T, D], where T = number of
                thresholds, D = number of detections
            `gtMatches`: matched ground truth boxes [T, G], where T = number
                of thresholds, G = number of ground truth
            `dtScores`: prediction scores [D] detection scores
            `gtIgnore`: ground truth boxes which should be ignored
                [G] indicate whether ground truth should be ignored
            `dtIgnore`: detections which should be ignored [T, D],
                indicate which detections should be ignored
    """
    dt_ind = np.argsort(-pred_scores, kind='mergesort')
    dt_ind = dt_ind[:max_detections]
    dt_scores = pred_scores[dt_ind]

    num_preds = len(dt_scores)

    gt_match = np.array([[]] * len(iou_thresholds))
    dt_match = np.zeros((len(iou_thresholds), num_preds))
    dt_ignore = np.zeros((len(iou_thresholds), num_preds))

    return {
        'dtMatches': dt_match,  # [T, D], where T = number of thresholds, D = number of detections
        'gtMatches': gt_match,  # [T, G], where T = number of thresholds, G = number of ground truth
        'dtScores': dt_scores,  # [D] detection scores
        'gtIgnore': np.array([]).reshape(-1),  # [G] indicate whether ground truth should be ignored
        'dtIgnore': dt_ignore,  # [T, D], indicate which detections should be ignored
    }


def _matching_no_pred(
        iou_thresholds: Sequence[float],
        gt_ignore: np.ndarray,
        ):
    """
    Matching result with no predictions

    Args:
        iou_thresholds: defined which IoU thresholds should be evaluated
        gt_ignore: specified if which ground truth boxes are not counted as
            true positives (detections which match theses boxes are not
            counted as false positives either); [G], G number of ground truth

    Returns:
        dict: computed matching
            `dtMatches`: matched detections [T, D], where T = number of
                thresholds, D = number of detections
            `gtMatches`: matched ground truth boxes [T, G], where T = number
                of thresholds, G = number of ground truth
            `dtScores`: prediction scores [D] detection scores
            `gtIgnore`: ground truth boxes which should be ignored
                [G] indicate whether ground truth should be ignored
            `dtIgnore`: detections which should be ignored [T, D],
                indicate which detections should be ignored
    """
    dt_scores = np.array([])
    dt_match = np.array([[]] * len(iou_thresholds))
    dt_ignore = np.array([[]] * len(iou_thresholds))

    n_gt = 0 if gt_ignore.size == 0 else gt_ignore.shape[0]
    gt_match = np.zeros((len(iou_thresholds), n_gt))

    return {
        'dtMatches': dt_match,  # [T, D], where T = number of thresholds, D = number of detections
        'gtMatches': gt_match,  # [T, G], where T = number of thresholds, G = number of ground truth
        'dtScores': dt_scores,  # [D] detection scores
        'gtIgnore': gt_ignore.reshape(-1),  # [G] indicate whether ground truth should be ignored
        'dtIgnore': dt_ignore,  # [T, D], indicate which detections should be ignored
    }


def _matching_single_image_single_class(
        iou_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        gt_ignore: np.ndarray,
        max_detections: int,
        iou_thresholds: Sequence[float],
        ) -> Dict[str, np.ndarray]:
    """
    Adapted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Args:
        iou_fn: compute overlap for each pair
        iou_thresholds: defined which IoU thresholds should be evaluated
        pred_boxes: predicted boxes from single batch; [D, dim * 2], D number
            of predictions
        pred_scores: predicted score for each bounding box; [D], D number of
            predictions
        gt_boxes: ground truth boxes; [G, dim * 2], G number of ground truth
        gt_ignore: specified if which ground truth boxes are not counted as
            true positives (detections which match theses boxes are not
            counted as false positives either); [G], G number of ground truth
        max_detections: maximum number of detections which should be evaluated

    Returns:
        dict: computed matching
            `dtMatches`: matched detections [T, D], where T = number of
                thresholds, D = number of detections
            `gtMatches`: matched ground truth boxes [T, G], where T = number
                of thresholds, G = number of ground truth
            `dtScores`: prediction scores [D] detection scores
            `gtIgnore`: ground truth boxes which should be ignored
                [G] indicate whether ground truth should be ignored
            `dtIgnore`: detections which should be ignored [T, D],
                indicate which detections should be ignored
    """
    # filter for max_detections highest scoring predictions to speed up computation
    dt_ind = np.argsort(-pred_scores, kind='mergesort')
    dt_ind = dt_ind[:max_detections]

    pred_boxes = pred_boxes[dt_ind]
    pred_scores = pred_scores[dt_ind]

    # sort ignored ground truth to last positions
    gt_ind = np.argsort(gt_ignore, kind='mergesort')
    gt_boxes = gt_boxes[gt_ind]
    gt_ignore = gt_ignore[gt_ind]

    # ious between sorted(!) predictions and ground truth
    ious = iou_fn(pred_boxes, gt_boxes)

    num_preds, num_gts = ious.shape[0], ious.shape[1]
    gt_match = np.zeros((len(iou_thresholds), num_gts))
    dt_match = np.zeros((len(iou_thresholds), num_preds))
    dt_ignore = np.zeros((len(iou_thresholds), num_preds))

    for tind, t in enumerate(iou_thresholds):
        for dind, _d in enumerate(pred_boxes):  # iterate detections starting from highest scoring one
            # information about best match so far (m=-1 -> unmatched)
            iou = min([t, 1-1e-10])
            m = -1

            for gind, _g in enumerate(gt_boxes):  # iterate ground truth
                # if this gt already matched, continue
                if gt_match[tind, gind] > 0:
                    continue

                # if dt matched to reg gt, and on ignore gt, stop
                if m > -1 and gt_ignore[m] == 0 and gt_ignore[gind] == 1:
                    break

                # continue to next gt unless better match made
                if ious[dind, gind] < iou:
                    continue

                # if match successful and best so far, store appropriately
                iou = ious[dind, gind]
                m = gind

            # if match made, store id of match for both dt and gt
            if m == -1:
                continue
            else:
                dt_ignore[tind, dind] = int(gt_ignore[m])
                dt_match[tind, dind] = 1
                gt_match[tind, m] = 1

    # store results for given image and category
    return {
            'dtMatches': dt_match,  # [T, D], where T = number of thresholds, D = number of detections
            'gtMatches': gt_match,  # [T, G], where T = number of thresholds, G = number of ground truth
            'dtScores': pred_scores,  # [D] detection scores
            'gtIgnore': gt_ignore.reshape(-1),  # [G] indicate whether ground truth should be ignored
            'dtIgnore': dt_ignore,  # [T, D], indicate which detections should be ignored
        }
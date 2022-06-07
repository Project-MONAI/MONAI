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
# Adapted from https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/evaluator/detection/coco.py
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
# Adapted from https://github.com/cocodataset/cocoapi
# which has the following license...
# https://github.com/cocodataset/cocoapi/blob/master/license.txt

# Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.

"""
This script is almost same with https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/evaluator/detection/coco.py
The changes include 1) code reformatting, 2) docstrings.
"""

import logging as logger
import time
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np


class COCOMetric:
    def __init__(
        self,
        classes: Sequence[str],
        iou_list: Sequence[float] = (0.1, 0.5, 0.75),
        iou_range: Sequence[float] = (0.1, 0.5, 0.05),
        max_detection: Sequence[int] = (1, 5, 100),
        per_class: bool = True,
        verbose: bool = True,
    ):
        """
        Class to compute COCO metrics
        Metrics computed includes,

        - mAP over the IoU range specified by `iou_range` at last value of `max_detection`
        - AP values at IoU thresholds specified by `iou_list` at last value of `max_detection`
        - AR over max detections thresholds defined by `max_detection` (over iou range)

        Args:
            classes (Sequence[str]): name of each class (index needs to correspond to predicted class indices!)
            iou_list (Sequence[float]): specific thresholds where ap is evaluated and saved
            iou_range (Sequence[float]): (start, stop, step) for mAP iou thresholds
            max_detection (Sequence[int]): maximum number of detections per image
            verbose (bool): log time needed for evaluation

        Example:

            .. code-block:: python

                from monai.data.box_utils import box_iou
                from monai.apps.detection.metrics.coco import COCOMetric
                from monai.apps.detection.metrics.matching import matching_batch
                # 3D example outputs of one image from detector
                val_outputs_all = [
                        {"boxes": torch.tensor([[1,1,1,3,4,5]],dtype=torch.float16),
                        "labels": torch.randint(3,(1,)),
                        "scores": torch.randn((1,)).absolute()},
                ]
                val_targets_all = [
                        {"boxes": torch.tensor([[1,1,1,2,6,4]],dtype=torch.float16),
                        "labels": torch.randint(3,(1,))},
                ]

                coco_metric = COCOMetric(
                    classes=['c0','c1','c2'], iou_list=[0.1], max_detection=[10]
                )
                results_metric = matching_batch(
                    iou_fn=box_iou,
                    iou_thresholds=coco_metric.iou_thresholds,
                    pred_boxes=[val_data_i["boxes"].numpy() for val_data_i in val_outputs_all],
                    pred_classes=[val_data_i["labels"].numpy() for val_data_i in val_outputs_all],
                    pred_scores=[val_data_i["scores"].numpy() for val_data_i in val_outputs_all],
                    gt_boxes=[val_data_i["boxes"].numpy() for val_data_i in val_targets_all],
                    gt_classes=[val_data_i["labels"].numpy() for val_data_i in val_targets_all],
                )
                val_metric_dict = coco_metric(results_metric)
                print(val_metric_dict)
        """
        self.verbose = verbose
        self.classes = classes
        self.per_class = per_class

        iou_list_np = np.array(iou_list)
        _iou_range = np.linspace(
            iou_range[0], iou_range[1], int(np.round((iou_range[1] - iou_range[0]) / iou_range[2])) + 1, endpoint=True
        )
        self.iou_thresholds = np.union1d(iou_list_np, _iou_range)
        self.iou_range = iou_range

        # get indices of iou values of ious range and ious list for later evaluation
        self.iou_list_idx = np.nonzero(iou_list_np[:, np.newaxis] == self.iou_thresholds[np.newaxis])[1]
        self.iou_range_idx = np.nonzero(_iou_range[:, np.newaxis] == self.iou_thresholds[np.newaxis])[1]

        if (
            not (self.iou_thresholds[self.iou_list_idx] == iou_list_np).all()
            or not (self.iou_thresholds[self.iou_range_idx] == _iou_range).all()
        ):
            raise ValueError(
                "Require self.iou_thresholds[self.iou_list_idx] == iou_list_np and "
                "self.iou_thresholds[self.iou_range_idx] == _iou_range."
            )

        self.recall_thresholds = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        self.max_detections = max_detection

    def __call__(self, *args, **kwargs) -> Tuple[Dict[str, float], Union[Dict[str, np.ndarray], None]]:
        """
        Compute metric. See :func:`compute` for more information.

        Args:
            *args: positional arguments passed to :func:`compute`
            **kwargs: keyword arguments passed to :func:`compute`

        Returns:
            Dict[str, float]: dictionary with scalar values for evaluation
            Dict[str, np.ndarray]: dictionary with arrays, e.g. for visualization of graphs
        """
        return self.compute(*args, **kwargs)

    def check_number_of_iou(self, *args) -> None:
        """
        Check if shape of input in first dimension is consistent with expected IoU values
        (assumes IoU dimension is the first dimension)

        Args:
            args: array like inputs with shape function
        """
        num_ious = len(self.get_iou_thresholds())
        for arg in args:
            if arg.shape[0] != num_ious:
                raise ValueError(
                    f"Require arg.shape[0] == len(self.get_iou_thresholds()). Got arg.shape[0]={arg.shape[0]}, "
                    f"self.get_iou_thresholds()={self.get_iou_thresholds()}."
                )

    def get_iou_thresholds(self) -> Sequence[float]:
        """
        Return IoU thresholds needed for this metric in an numpy array

        Returns:
            Sequence[float]: IoU thresholds [M], M is the number of thresholds
        """
        return list(self.iou_thresholds)

    def compute(self, results_list: List[Dict[int, Dict[str, np.ndarray]]]) -> Tuple[Dict[str, float], None]:
        """
        Compute COCO metrics

        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with results per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.

                - `dtMatches`: matched detections [T, D], where T = number of
                  thresholds, D = number of detections
                - `gtMatches`: matched ground truth boxes [T, G], where T = number
                  of thresholds, G = number of ground truth
                - `dtScores`: prediction scores [D] detection scores
                - `gtIgnore`: ground truth boxes which should be ignored
                  [G] indicate whether ground truth should be ignored
                - `dtIgnore`: detections which should be ignored [T, D],
                  indicate which detections should be ignored

        Returns:
            Dict[str, float], dictionary with coco metrics
        """
        if self.verbose:
            logger.info("Start COCO metric computation...")
            tic = time.time()

        dataset_statistics = self._compute_statistics(results_list=results_list)  # Dict[str, Union[np.ndarray, List]]

        if self.verbose:
            toc = time.time()
            logger.info(f"Statistics for COCO metrics finished (t={(toc - tic):0.2f}s).")

        results = {}
        results.update(self._compute_ap(dataset_statistics))
        results.update(self._compute_ar(dataset_statistics))

        if self.verbose:
            toc = time.time()
            logger.info(f"COCO metrics computed in t={(toc - tic):0.2f}s.")
        return results, None

    def _compute_ap(self, dataset_statistics: Dict[str, Union[np.ndarray, List]]) -> Dict[str, float]:
        """
        Compute AP metrics

        Args:
            dataset_statistics (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.

                - `dtMatches`: matched detections [T, D], where T = number of
                  thresholds, D = number of detections
                - `gtMatches`: matched ground truth boxes [T, G], where T = number
                  of thresholds, G = number of ground truth
                - `dtScores`: prediction scores [D] detection scores
                - `gtIgnore`: ground truth boxes which should be ignored
                  [G] indicate whether ground truth should be ignored
                - `dtIgnore`: detections which should be ignored [T, D],
                  indicate which detections should be ignored
        """
        results = {}
        if self.iou_range:  # mAP
            key = (
                f"mAP_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_"
                f"MaxDet_{self.max_detections[-1]}"
            )
            results[key] = self._select_ap(dataset_statistics, iou_idx=self.iou_range_idx, max_det_idx=-1)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (
                        f"{cls_str}_"
                        f"mAP_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_"
                        f"MaxDet_{self.max_detections[-1]}"
                    )
                    results[key] = self._select_ap(
                        dataset_statistics, iou_idx=self.iou_range_idx, cls_idx=cls_idx, max_det_idx=-1
                    )

        for idx in self.iou_list_idx:  # AP@IoU
            key = f"AP_IoU_{self.iou_thresholds[idx]:.2f}_MaxDet_{self.max_detections[-1]}"
            results[key] = self._select_ap(dataset_statistics, iou_idx=[idx], max_det_idx=-1)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = f"{cls_str}_" f"AP_IoU_{self.iou_thresholds[idx]:.2f}_" f"MaxDet_{self.max_detections[-1]}"
                    results[key] = self._select_ap(dataset_statistics, iou_idx=[idx], cls_idx=cls_idx, max_det_idx=-1)
        return results

    def _compute_ar(self, dataset_statistics: Dict[str, Union[np.ndarray, List]]) -> Dict[str, float]:
        """
        Compute AR metrics

        Args:
            dataset_statistics (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.

                - `dtMatches`: matched detections [T, D], where T = number of
                  thresholds, D = number of detections
                - `gtMatches`: matched ground truth boxes [T, G], where T = number
                  of thresholds, G = number of ground truth
                - `dtScores`: prediction scores [D] detection scores
                - `gtIgnore`: ground truth boxes which should be ignored
                  [G] indicate whether ground truth should be ignored
                - `dtIgnore`: detections which should be ignored [T, D],
                  indicate which detections should be ignored
        """
        results = {}
        for max_det_idx, max_det in enumerate(self.max_detections):  # mAR
            key = f"mAR_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_MaxDet_{max_det}"
            results[key] = self._select_ar(dataset_statistics, max_det_idx=max_det_idx)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (
                        f"{cls_str}_"
                        f"mAR_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_"
                        f"MaxDet_{max_det}"
                    )
                    results[key] = self._select_ar(dataset_statistics, cls_idx=cls_idx, max_det_idx=max_det_idx)

        for idx in self.iou_list_idx:  # AR@IoU
            key = f"AR_IoU_{self.iou_thresholds[idx]:.2f}_MaxDet_{self.max_detections[-1]}"
            results[key] = self._select_ar(dataset_statistics, iou_idx=idx, max_det_idx=-1)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = f"{cls_str}_" f"AR_IoU_{self.iou_thresholds[idx]:.2f}_" f"MaxDet_{self.max_detections[-1]}"
                    results[key] = self._select_ar(dataset_statistics, iou_idx=idx, cls_idx=cls_idx, max_det_idx=-1)
        return results

    @staticmethod
    def _select_ap(
        dataset_statistics: dict,
        iou_idx: Union[int, List[int], np.ndarray, None] = None,
        cls_idx: Union[int, Sequence[int], None] = None,
        max_det_idx: int = -1,
    ) -> float:
        """
        Compute average precision

        Args:
            dataset_statistics (dict): computed statistics over dataset

                - `counts`: Number of thresholds, Number recall thresholds, Number of classes, Number of max
                  detection thresholds
                - `recall`: Computed recall values [num_iou_th, num_classes, num_max_detections]
                - `precision`: Precision values at specified recall thresholds
                  [num_iou_th, num_recall_th, num_classes, num_max_detections]
                - `scores`: Scores corresponding to specified recall thresholds
                  [num_iou_th, num_recall_th, num_classes, num_max_detections]
            iou_idx: index of IoU values to select for evaluation(if None, all values are used)
            cls_idx: class indices to select, if None all classes will be selected
            max_det_idx (int): index to select max detection threshold from data

        Returns:
            np.ndarray: AP value
        """
        prec = dataset_statistics["precision"]
        if iou_idx is not None:
            prec = prec[iou_idx]
        if cls_idx is not None:
            prec = prec[..., cls_idx, :]
        prec = prec[..., max_det_idx]
        return float(np.mean(prec))

    @staticmethod
    def _select_ar(
        dataset_statistics: dict,
        iou_idx: Union[int, Sequence[int], None] = None,
        cls_idx: Union[int, Sequence[int], None] = None,
        max_det_idx: int = -1,
    ) -> float:
        """
        Compute average recall

        Args:
            dataset_statistics (dict): computed statistics over dataset

                - `counts`: Number of thresholds, Number recall thresholds, Number of classes, Number of max
                  detection thresholds
                - `recall`: Computed recall values [num_iou_th, num_classes, num_max_detections]
                - `precision`: Precision values at specified recall thresholds
                  [num_iou_th, num_recall_th, num_classes, num_max_detections]
                - `scores`: Scores corresponding to specified recall thresholds
                  [num_iou_th, num_recall_th, num_classes, num_max_detections]
            iou_idx: index of IoU values to select for evaluation(if None, all values are used)
            cls_idx: class indices to select, if None all classes will be selected
            max_det_idx (int): index to select max detection threshold from data

        Returns:
            np.ndarray: recall value
        """
        rec = dataset_statistics["recall"]
        if iou_idx is not None:
            rec = rec[iou_idx]
        if cls_idx is not None:
            rec = rec[..., cls_idx, :]
        rec = rec[..., max_det_idx]

        if len(rec[rec > -1]) == 0:
            return -1.0

        return float(np.mean(rec[rec > -1]))

    def _compute_statistics(
        self, results_list: List[Dict[int, Dict[str, np.ndarray]]]
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Compute statistics needed for COCO metrics (mAP, AP of individual classes, mAP@IoU_Thresholds, AR)
        Adapted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per cateory (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.

                - `dtMatches`: matched detections [T, D], where T = number of
                  thresholds, D = number of detections
                - `gtMatches`: matched ground truth boxes [T, G], where T = number
                  of thresholds, G = number of ground truth
                - `dtScores`: prediction scores [D] detection scores
                - `gtIgnore`: ground truth boxes which should be ignored
                  [G] indicate whether ground truth should be ignored
                - `dtIgnore`: detections which should be ignored [T, D],
                  indicate which detections should be ignored

        Returns:
            dict: computed statistics over dataset
                - `counts`: Number of thresholds, Number recall thresholds, Number of classes, Number of max
                  detection thresholds
                - `recall`: Computed recall values [num_iou_th, num_classes, num_max_detections]
                - `precision`: Precision values at specified recall thresholds
                  [num_iou_th, num_recall_th, num_classes, num_max_detections]
                - `scores`: Scores corresponding to specified recall thresholds
                  [num_iou_th, num_recall_th, num_classes, num_max_detections]
        """
        num_iou_th = len(self.iou_thresholds)
        num_recall_th = len(self.recall_thresholds)
        num_classes = len(self.classes)
        num_max_detections = len(self.max_detections)

        # -1 for the precision of absent categories
        precision = -np.ones((num_iou_th, num_recall_th, num_classes, num_max_detections))
        recall = -np.ones((num_iou_th, num_classes, num_max_detections))
        scores = -np.ones((num_iou_th, num_recall_th, num_classes, num_max_detections))

        for cls_idx, cls_i in enumerate(self.classes):  # for each class
            for max_det_idx, max_det in enumerate(self.max_detections):  # for each maximum number of detections
                results = [r[cls_idx] for r in results_list if cls_idx in r]  # len is num_images

                if len(results) == 0:
                    logger.warning(f"WARNING, no results found for coco metric for class {cls_i}")
                    continue

                dt_scores = np.concatenate([r["dtScores"][0:max_det] for r in results])
                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dt_scores, kind="mergesort")
                dt_scores_sorted = dt_scores[inds]

                # r['dtMatches'] [T, R], where R = sum(all detections)
                dt_matches = np.concatenate([r["dtMatches"][:, 0:max_det] for r in results], axis=1)[:, inds]
                dt_ignores = np.concatenate([r["dtIgnore"][:, 0:max_det] for r in results], axis=1)[:, inds]
                self.check_number_of_iou(dt_matches, dt_ignores)
                gt_ignore = np.concatenate([r["gtIgnore"] for r in results])
                num_gt = np.count_nonzero(gt_ignore == 0)  # number of ground truth boxes (non ignored)
                if num_gt == 0:
                    logger.warning(f"WARNING, no gt found for coco metric for class {cls_i}")
                    continue

                # ignore cases need to be handled differently for tp and fp
                tps = np.logical_and(dt_matches, np.logical_not(dt_ignores))
                fps = np.logical_and(np.logical_not(dt_matches), np.logical_not(dt_ignores))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float32)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float32)

                for th_ind, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):  # for each threshold th_ind
                    tp, fp = np.array(tp), np.array(fp)
                    r, p, s = _compute_stats_single_threshold(tp, fp, dt_scores_sorted, self.recall_thresholds, num_gt)
                    recall[th_ind, cls_idx, max_det_idx] = r
                    precision[th_ind, :, cls_idx, max_det_idx] = p
                    # corresponding score thresholds for recall steps
                    scores[th_ind, :, cls_idx, max_det_idx] = s

        return {
            "counts": [num_iou_th, num_recall_th, num_classes, num_max_detections],  # [4]
            "recall": recall,  # [num_iou_th, num_classes, num_max_detections]
            "precision": precision,  # [num_iou_th, num_recall_th, num_classes, num_max_detections]
            "scores": scores,  # [num_iou_th, num_recall_th, num_classes, num_max_detections]
        }


def _compute_stats_single_threshold(
    tp: np.ndarray,
    fp: np.ndarray,
    dt_scores_sorted: np.ndarray,
    recall_thresholds: Union[np.ndarray, Sequence[float]],
    num_gt: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute recall value, precision curve and scores thresholds
    Adapted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Args:
        tp (np.ndarray): cumsum over true positives [R], R is the number of detections
        fp (np.ndarray): cumsum over false positives [R], R is the number of detections
        dt_scores_sorted (np.ndarray): sorted (descending) scores [R], R is the number of detections
        recall_thresholds (Sequence[float]): recall thresholds which should be evaluated
        num_gt (int): number of ground truth bounding boxes (excluding boxes which are ignored)

    Returns:
        - float, overall recall for given IoU value
        - np.ndarray, precision values at defined recall values
          [RTH], where RTH is the number of recall thresholds
        - np.ndarray, prediction scores corresponding to recall values
          [RTH], where RTH is the number of recall thresholds
    """
    num_recall_th = len(recall_thresholds)

    rc = tp / num_gt
    # np.spacing(1) is the smallest representable epsilon with float
    pr = tp / (fp + tp + np.spacing(1))

    if len(tp):
        recall = rc[-1]
    else:
        # no prediction
        recall = 0

    # array where precision values nearest to given recall th are saved
    precision = np.zeros((num_recall_th,))
    # save scores for corresponding recall value in here
    th_scores = np.zeros((num_recall_th,))
    # numpy is slow without cython optimization for accessing elements
    # use python array gets significant speed improvement
    pr = pr.tolist()
    precision = precision.tolist()

    # smooth precision curve (create box shape)
    for i in range(len(tp) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]

    # get indices to nearest given recall threshold (nn interpolation!)
    inds = np.searchsorted(rc, recall_thresholds, side="left")
    try:
        for save_idx, array_index in enumerate(inds):
            precision[save_idx] = pr[array_index]
            th_scores[save_idx] = dt_scores_sorted[array_index]
    except BaseException as err:
        pass

    return recall, np.array(precision), np.array(th_scores)

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
# Adapted from https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/evaluator/abstract.py
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

"""
This script is same with https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/evaluator/abstract.py
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

__all__ = ["AbstractEvaluator", "AbstractMetric", "DetectionMetric"]


class AbstractEvaluator(ABC):
    @abstractmethod
    def run_online_evaluation(self, *args, **kwargs):
        """
        Compute necessary values per batch for later evaluation
        """
        raise NotImplementedError

    @abstractmethod
    def finish_online_evaluation(self, *args, **kwargs):
        """
        Accumulate results from batches and compute metrics
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Reset internal state of evaluator
        """
        raise NotImplementedError


class AbstractMetric(ABC):
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

    @abstractmethod
    def compute(
        self, results_list: List[Dict[int, Dict[str, np.ndarray]]]
    ) -> Tuple[Dict[str, float], Union[Dict[str, np.ndarray], None]]:
        """
        Compute metric

        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, G], where T = number of thresholds, G = number of ground truth
                `gtMatches`: matched ground truth boxes [T, D], where T = number of thresholds,
                    D = number of detections
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth
                    should be ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored

        Returns:
            Dict[str, float]: dictionary with scalar values for evaluation
            Dict[str, np.ndarray]: dictionary with arrays, e.g. for visualization of graphs
        """
        raise NotImplementedError


class DetectionMetric(AbstractMetric):
    @abstractmethod
    def get_iou_thresholds(self) -> Sequence[float]:
        """
        Return IoU thresholds needed for this metric in an numpy array

        Returns:
            Sequence[float]: IoU thresholds; [M], M is the number of thresholds
        """
        raise NotImplementedError

    def check_number_of_iou(self, *args) -> None:
        """
        Check if shape of input in first dimension is consistent with expected IoU values
        (assumes IoU dimension is the first dimension)

        Args:
            args: array like inputs with shape function
        """
        num_ious = len(self.get_iou_thresholds())
        for arg in args:
            assert arg.shape[0] == num_ious

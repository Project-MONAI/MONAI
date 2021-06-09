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

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import torch

from monai.config import TensorOrList
from monai.utils import evenly_divisible_all_gather


class Metric(ABC):
    """
    Base class of all Metrics inerface.
    `__call__` is designed to execute metric computation, and `reset` is designed to reset
    all the states and environments for next round.

    """

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        API to execute the metric computation.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def reset(self, *args: Any, **kwds: Any) -> Any:
        """
        API to reset all the states and environments to next round computation.

        """
        pass

    def sync(self, *args: Any, **kwds: Any) -> Any:
        """
        API to sync data of all the ranks in distributed computation parallel.

        """
        pass


class CumulativeMetric(Metric):
    """
    Base class of cumulative Metrics interface.
    `__call__` is supposed to compute independent logic for several samples of `y_pred` and `y`(optional).
    Ususally, subclass only needs to implement the `_apply` function for computation process.
    And `reduce` is supposed to execute reduction for the final result, it can be used for 1 batch data
    or for the accumulated overall data.

    """

    def __call__(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):
        """
        Execute basic computation for model prediction and ground truth.
        It can support  both `list of channel-first Tensor` and `batch-first Tensor`.
        And users can execute on every batch of data, then accumulate the results, or
        accumulate the original `y_pred` and `y`, then execute on the accumulated data.

        Args:
            y_pred: the model prediction data to compute, must be a list of `channel-first` Tensor
                or a `batch-first` Tensor.
            y: the ground truth to compute, must be a list of `channel-first` Tensor
                or a `batch-first` Tensor.

        """
        ret: TensorOrList
        if isinstance(y_pred, (list, tuple)) or isinstance(y, (list, tuple)):
            # if y_pred or y is a list of channel-first data, add batch dim and compute metric
            ret_: List[torch.Tensor] = self._compute_list(y_pred, y)
            # concat the list of results
            if isinstance(ret_[0], torch.Tensor):
                ret = torch.cat(ret_, dim=0)
            elif isinstance(ret_[0], (list, tuple)) and all([isinstance(i, torch.Tensor) for i in ret_[0]]):
                # if _compute() returned not only 1 Tensor, concat them separately
                ret = [torch.cat([k[i] for k in ret_], dim=0) for i in range(len(ret_[0]))]
            else:
                # if not expected data type, return raw results directly
                ret = ret_
        elif isinstance(y_pred, torch.Tensor):
            y_ = y.detach() if y is not None and isinstance(y, torch.Tensor) else None
            ret = self._compute(y_pred.detach(), y_)
        else:
            raise ValueError("y_pred or y must be a list of `channel-first` Tensors or a `batch-first` Tensor.")

        return ret

    def _compute_list(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):
        """
        Excute the computation for every item of a list.
        Subclass may enhance the operation with multi-threads to accelerate.

        """
        if y is not None:
            return [self._compute(p_.detach().unsqueeze(0), y_.detach().unsqueeze(0)) for p_, y_ in zip(y_pred, y)]
        else:
            return [self._compute(p_.detach().unsqueeze(0), None) for p_ in y_pred]

    @abstractmethod
    def _compute(self, y_pred: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Actual computation logic of the metric, input data should be `batch-first` Tensor.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def aggregate(self, *args: Any, **kwds: Any) -> Any:
        """
        Aggregate the metric results. Users can call it for the batch data of every iteration
        or accumulte the results of every iteration and call it for the final output.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class IterationMetric(CumulativeMetric):
    def __init__(self) -> None:
        super().__init__()
        self._scores: List = []
        self._synced_scores: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._scores = []
        self._synced_scores = None

    def sync(self) -> torch.Tensor:
        scores = torch.cat(self._scores, dim=0)
        # all gather across all processes
        self._synced_scores = evenly_divisible_all_gather(data=scores, concat=True)

        return self._synced_scores

    def __call__(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):
        score = super().__call__(y_pred=y_pred, y=y)
        self._scores.append(score)

        return score


class EpochMetric(CumulativeMetric):
    def __init__(self) -> None:
        super().__init__()
        self._y_pred: List = []
        self._y: List = []
        self._synced_y_pred: Optional[torch.Tensor] = None
        self._synced_y: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._y_pred = []
        self._y = []
        self._synced_y_pred = None
        self._synced_y = None

    def sync(self) -> Tuple[torch.Tensor, torch.Tensor]:
        y_pred = torch.cat(self._y_pred, dim=0)
        y = torch.cat(self._y, dim=0) if len(self._y) > 0 else None
        # all gather across all processes
        self._synced_y_pred = evenly_divisible_all_gather(data=y_pred, concat=True)
        self._synced_y = evenly_divisible_all_gather(data=y, concat=True) if y is not None else None

        return self._synced_y_pred, self._synced_y

    def __call__(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):
        y_pred, y = super().__call__(y_pred=y_pred, y=y)
        self._y_pred.append(y_pred)
        if y is not None:
            self._y.append(y)

        return y_pred, y

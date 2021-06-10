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
from typing import Any, List, Optional

import torch

from monai.config import TensorOrList
from monai.utils import evenly_divisible_all_gather


class Metric(ABC):
    """
    Base class of all Metrics inerface.
    `__call__` is designed to execute metric computation.

    """

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any):
        """
        API to execute the metric computation.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class IterationMetric(Metric):
    """
    Base class of Metrics interface for computation on a batch of tensors, usually the data of 1 iteration.
    `__call__` is supposed to compute independent logic for several samples of `y_pred` and `y`(optional).
    Ususally, subclass only needs to implement the `_compute_tensor` function for computation process.

    """

    def __call__(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):  # type: ignore
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
                # if _compute_tensor() returned not only 1 Tensor, concat them separately
                ret = [torch.cat([k[i] for k in ret_], dim=0) for i in range(len(ret_[0]))]
            else:
                # if not expected data type, return raw results directly
                ret = ret_
        elif isinstance(y_pred, torch.Tensor):
            y_ = y.detach() if y is not None and isinstance(y, torch.Tensor) else None
            ret = self._compute_tensor(y_pred.detach(), y_)
        else:
            raise ValueError("y_pred or y must be a list of `channel-first` Tensors or a `batch-first` Tensor.")

        return ret

    def _compute_list(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):
        """
        Excute the computation for every item of a list.
        Subclass may enhance the operation with multi-threads to accelerate.

        """
        if y is not None:
            return [
                self._compute_tensor(p_.detach().unsqueeze(0), y_.detach().unsqueeze(0)) for p_, y_ in zip(y_pred, y)
            ]
        else:
            return [self._compute_tensor(p_.detach().unsqueeze(0), None) for p_ in y_pred]

    @abstractmethod
    def _compute_tensor(self, y_pred: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        computation logic for tensors, input data should be `batch-first` Tensor.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class Cumulative:
    """
    Utility class for the typical cumulative computation process based on PyTorch Tensor data.
    Execute the steps referring to below examples::

        cum = Cumulative(buffer_count=2)
        cum.add(x, y)
        cum.add(a, b)
        cum.add(c, d)
        cum.agrregate()
        result = cum.get_synced_tensors()
        cum.reset()

    Args:
        buffer_num: the number of buffers to create to cumulate PyTorch Tensors,
            usually every Tensor should map to a buffer.

    """

    def __init__(self, buffer_num: int = 1):
        self.buffer_num = buffer_num
        self.reset()

    def reset(self):
        """
        Reset buffers to cumulate tensors and the synced results.

        """
        self._buffers: List[List[torch.Tensor]] = [[] for _ in range(self.buffer_num)]
        self._synced_tensors: List[Optional[torch.Tensor]] = [None for _ in range(self.buffer_num)]

    def add(self, *data: torch.Tensor):
        """
        Add samples to the cumulative buffers.

        Args:
            data: list of input tensor, make sure the input data order is always the same.
                every item of data will be added to the corresponding buffer.

        """
        for i, d in enumerate(data):
            if not isinstance(d, torch.Tensor):
                raise ValueError(f"the data to cumulate in a buffer must be PyTorch Tensor, but got: {type(d)}.")
            self._buffers[i].append(d)

    def aggregate(self):
        """
        Aggregate final results based on buffers.
        This base class only syncs the data from distributed ranks, subclasses should implement more logic.

        """
        self._sync()

    def _sync(self):
        """
        Sync up distributed data when aggregating.

        """
        self._synced_tensors = [evenly_divisible_all_gather(torch.cat(b, dim=0), concat=True) for b in self._buffers]

    def get_synced_tensors(self):
        """
        Get the synced tensor list for other use cases.
        For example, generate the metrics report based on the raw metric details.

        """
        return self._synced_tensors[0] if len(self._synced_tensors) == 1 else self._synced_tensors


class CumulativeIterationMetric(Cumulative, IterationMetric):
    """
    Base class of cumulative metric for batch data.
    Typically, it computes some intermediate results for every iteration, cumulates in buffers,
    then syncs across all the distributed ranks and aggregates for the final result when epoch completed.

    """

    def __call__(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):  # type: ignore
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
        ret = super().__call__(y_pred=y_pred, y=y)
        if isinstance(ret, (tuple, list)):
            self.add(*ret)
        else:
            self.add(ret)

        return ret

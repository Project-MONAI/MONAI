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
    Base class of all Metrics interface.
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
    Usually, subclass only needs to implement the `_compute_tensor` function for computation process.
    The input data shape should be `list of channel-first tensors` or a `batch-first tensor`.

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
            ret = self._compute_list(y_pred, y)
        elif isinstance(y_pred, torch.Tensor):
            y_ = y.detach() if y is not None and isinstance(y, torch.Tensor) else None
            ret = self._compute_tensor(y_pred.detach(), y_)
        else:
            raise ValueError("y_pred or y must be a list of `channel-first` Tensors or a `batch-first` Tensor.")

        return ret

    def _compute_list(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):
        """
        Excute the computation for the y_pred and y items of a iteration, the data is in the list shape.
        Will concat the results to guarantee the output shape of ret is BCHW[D], otherwise it's list of batch-first,
        which is against our principle that data in metrics should be BCHW[D] or list of channel-first.
        Note: subclass may enhance the operation with multi-threads to accelerate.

        """
        ret: TensorOrList
        if y is not None:
            ret = [self._compute_tensor(p.detach().unsqueeze(0), y_.detach().unsqueeze(0)) for p, y_ in zip(y_pred, y)]
        else:
            ret = [self._compute_tensor(p_.detach().unsqueeze(0), None) for p_ in y_pred]
        # concat the list of results
        if isinstance(ret[0], torch.Tensor):
            ret = torch.cat(ret, dim=0)
        elif isinstance(ret[0], (list, tuple)) and all(isinstance(i, torch.Tensor) for i in ret[0]):
            # if _compute_tensor() returned not only 1 Tensor, concat them separately
            ret = [torch.cat([k[i] for k in ret], dim=0) for i in range(len(ret[0]))]

        return ret

    @abstractmethod
    def _compute_tensor(self, y_pred: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        computation logic for the y_pred and y of a iteration, the data should be `batch-first` Tensors.
        Every subclass metric should implement its own computation logic according to its algorithm.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class Cumulative(ABC):
    """
    Utility class for the typical cumulative computation process based on PyTorch Tensors.
    It cumulates tensors in the buffer, then sync across distributed ranks and aggregate.

    To speed up computation with multi-processing, PyTorch programs usually split data to distributed ranks
    by `DistributedSampler` before an epoch, every rank then computes only based on its own data part and
    `add` to the buffers in its process. Eventually, sync the values of all ranks to compute the final results.

    Note: the data list should have the same length every time calling `add()` in a round,
    it will automatically create buffers according to the length of data list.

    Typically, this class is expected to execute the steps referring to below examples::

        cum = Cumulative()
        cum.add(x, y)
        cum.add(a, b)
        cum.add(c, d)
        cum.aggregate()
        result = cum.get_buffer()
        cum.reset()

    """

    def __init__(self):
        self.buffer_num: int = 0
        self._buffers: Optional[List[List[torch.Tensor]]] = None
        self._synced_tensors: Optional[List[Optional[torch.Tensor]]] = None
        self._synced: bool = False

    def reset(self):
        """
        Reset the buffers for cumulative tensors and the synced results.

        """
        self._buffers = None
        self._synced_tensors = None
        self._synced = False

    def add(self, *data: torch.Tensor):
        """
        Add samples to the cumulative buffers.

        Args:
            data: list of input tensor, make sure the input data order is always the same in a round.
                every item of data will be added to the corresponding buffer.

        """
        data_len = len(data)
        if self._buffers is None:
            self._buffers = [[] for _ in range(data_len)]
        elif len(self._buffers) != data_len:
            raise ValueError(f"data length: {data_len} doesn't match buffers length: {len(self._buffers)}.")
        if self._synced_tensors is None:
            self._synced_tensors = [None for _ in range(data_len)]

        for i, d in enumerate(data):
            if not isinstance(d, torch.Tensor):
                raise ValueError(f"the data to cumulate in a buffer must be PyTorch Tensor, but got: {type(d)}.")
            self._buffers[i].append(d)
        self._synced = False

    @abstractmethod
    def aggregate(self, *args: Any, **kwds: Any):
        """
        Aggregate final results based on the buffers.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def _sync(self):
        """
        All gather the buffers across distributed ranks for aggregating.
        Every buffer will be concatenated as a PyTorch Tensor.

        """
        self._synced_tensors = [evenly_divisible_all_gather(torch.cat(b, dim=0), concat=True) for b in self._buffers]
        self._synced = True

    def get_buffer(self):
        """
        Get the synced buffers list.
        A typical usage is to generate the metrics report based on the raw metric details.

        """
        if not self._synced:
            self._sync()
        return self._synced_tensors[0] if len(self._synced_tensors) == 1 else self._synced_tensors


class CumulativeIterationMetric(Cumulative, IterationMetric):
    """
    Base class of cumulative metric which computes on batch data of every iteration and aggregate.
    Typically, it computes some intermediate results for every iteration, cumulates in buffers,
    then syncs across all the distributed ranks and aggregates for the final result when epoch completed.

    """

    def __call__(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):  # type: ignore
        """
        Execute basic computation for model prediction and ground truth.
        It can support  both `list of channel-first Tensor` and `batch-first Tensor`.
        Users call this API to execute computation on every batch of data, then accumulate the results,
        or accumulate the original `y_pred` and `y`, then execute on the accumulated data.

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

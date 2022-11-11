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

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch

from monai.config import TensorOrList
from monai.utils import convert_data_type, evenly_divisible_all_gather

__all__ = ["Metric", "IterationMetric", "Cumulative", "CumulativeIterationMetric"]


class Metric(ABC):
    """
    Base class for metric computation for evaluating the performance of a model.
    `__call__` is designed to execute the computation.

    """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any):
        """
        This method should take raw model outputs as inputs, and return values that measure the models' quality.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class IterationMetric(Metric):
    """
    Base class for metrics computation at the iteration level, that is, on a min-batch of samples
    usually using the model outcome of one iteration.

    `__call__` is designed to handle `y_pred` and `y` (optional) in torch tensors or a list/tuple of tensors.

    Subclasses typically implement the `_compute_tensor` function for the actual tensor computation logic.
    """

    def __call__(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):
        """
        Execute basic computation for model prediction `y_pred` and ground truth `y` (optional).
        It supports inputs of a list of "channel-first" Tensor and a "batch-first" Tensor.

        Args:
            y_pred: the raw model prediction data at one iteration, must be a list of `channel-first` Tensor
                or a `batch-first` Tensor.
            y: the ground truth to compute, must be a list of `channel-first` Tensor
                or a `batch-first` Tensor.

        Returns:
            The computed metric values at the iteration level.
            The output shape could be a `batch-first` tensor or a list of `batch-first` tensors.
            When it's a list of tensors, each item in the list can represent a specific type of metric.

        """
        ret: TensorOrList
        # handling a list of channel-first data
        if isinstance(y_pred, (list, tuple)) or isinstance(y, (list, tuple)):
            return self._compute_list(y_pred, y)
        # handling a single batch-first data
        if isinstance(y_pred, torch.Tensor):
            y_ = y.detach() if isinstance(y, torch.Tensor) else None
            return self._compute_tensor(y_pred.detach(), y_)
        raise ValueError("y_pred or y must be a list/tuple of `channel-first` Tensors or a `batch-first` Tensor.")

    def _compute_list(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):
        """
        Execute the metric computation for `y_pred` and `y` in a list of "channel-first" tensors.

        The return value is a "batch-first" tensor, or a list of "batch-first" tensors.
        When it's a list of tensors, each item in the list can represent a specific type of metric values.

        For example, `self._compute_tensor` may be implemented as returning a list of `batch_size` items,
        where each item is a tuple of three values `tp`, `fp`, `fn` for true positives, false positives,
        and false negatives respectively. This function will return a list of three items,
        (`tp_batched`, `fp_batched`, `fn_batched`), where each item is a `batch_size`-length tensor.

        Note: subclass may enhance the operation to have multi-thread support.
        """
        if y is not None:
            ret = [self._compute_tensor(p.detach().unsqueeze(0), y_.detach().unsqueeze(0)) for p, y_ in zip(y_pred, y)]
        else:
            ret = [self._compute_tensor(p_.detach().unsqueeze(0), None) for p_ in y_pred]

        # concat the list of results (e.g. a batch of evaluation scores)
        if isinstance(ret[0], torch.Tensor):
            return torch.cat(ret, dim=0)
        # the result is a list of sequence of tensors (e.g. a batch of multi-class results)
        if isinstance(ret[0], (list, tuple)) and all(isinstance(i, torch.Tensor) for i in ret[0]):
            return [torch.cat(batch_i, dim=0) for batch_i in zip(*ret)]
        return ret

    @abstractmethod
    def _compute_tensor(self, y_pred: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Computation logic for `y_pred` and `y` of an iteration, the data should be "batch-first" Tensors.
        A subclass should implement its own computation logic.
        The return value is usually a "batch_first" tensor, or a list of "batch_first" tensors.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class Cumulative:
    """
    Utility class for the typical cumulative computation process based on PyTorch Tensors.
    It provides interfaces to accumulate values in the local buffers, synchronize buffers across distributed nodes,
    and aggregate the buffered values.

    In multi-processing, PyTorch programs usually distribute data to multiple nodes. Each node runs with a subset
    of the data, adds values to its local buffers. Calling `get_buffer` could gather all the results and
    `aggregate` can further handle the results to generate the final outcomes.

    Users can implement their own `aggregate` method to handle the results,
    using `get_buffer` to get the buffered contents.

    Note: the data list should have the same length every time calling `add()` in a round,
    it will automatically create buffers according to the length of data list.

    Typically, this class is expected to execute the following steps:

    .. code-block:: python

        from monai.metrics import Cumulative

        c = Cumulative()
        c.append(1)  # adds a value
        c.extend([2, 3])  # adds a batch of values
        c.extend([4, 5, 6])  # adds a batch of values
        print(c.get_buffer())  # tensor([1, 2, 3, 4, 5, 6])
        print(len(c))  # 6
        c.reset()
        print(len(c))  # 0

    The following is an example of maintaining two internal buffers:

    .. code-block:: python

        from monai.metrics import Cumulative

        c = Cumulative()
        c.append(1, 2)  # adds a value to two buffers respectively
        c.extend([3, 4], [5, 6])  # adds batches of values
        print(c.get_buffer())  # [tensor([1, 3, 4]), tensor([2, 5, 6])]
        print(len(c))

    The following is an example of extending with variable length data:

    .. code-block:: python

        import torch
        from monai.metrics import Cumulative

        c = Cumulative()
        c.extend(torch.zeros((8, 2)), torch.zeros((6, 2)))  # adds batches
        c.append(torch.zeros((2, )))  # adds a value
        print(c.get_buffer())  # [torch.zeros((9, 2)), torch.zeros((6, 2))]
        print(len(c))

    """

    def __init__(self):
        """
        Initialize the internal buffers.
        `self._buffers` are local buffers, they are not usually used directly.
        `self._sync_buffers` are the buffers with all the results across all the nodes.
        """
        self._buffers: Optional[List[List[torch.Tensor]]] = None
        self._synced_tensors: Optional[List[Optional[torch.Tensor]]] = None
        self._synced: bool = False
        self.reset()

    def reset(self):
        """
        Reset the buffers for cumulative tensors and the synced results.

        """
        self._buffers = None
        self._synced_tensors = None
        self._synced = False

    def extend(self, *data) -> None:
        """
        Extend the local buffers with new ("batch-first") data.
        A buffer will be allocated for each `data` item.
        Compared with `self.append`, this method adds a "batch" of data to the local buffers.

        Args:
            data: each item can be a "batch-first" tensor or a list of "channel-first" tensors.
                they will be concatenated at the 0-th dimension when `get_buffer()` is called.
        """
        if self._buffers is None:
            self._buffers = [[] for _ in data]
        for b, d in zip(self._buffers, data):
            # converting to pytorch tensors so that we can use the distributed API
            d_t, *_ = convert_data_type(d, output_type=torch.Tensor, wrap_sequence=True)
            try:  # d_t must be a mini-batch of values
                b.extend([x[0] for x in torch.split(d_t, 1, dim=0)])
            except (AttributeError, IndexError, RuntimeError) as e:
                raise TypeError(
                    f"{e}. `data` should be a batch-first tensor or"
                    f" a list of channel-first tensors, got {type(d_t)}"
                ) from e
        self._synced = False

    def append(self, *data) -> None:
        """
        Add samples to the local cumulative buffers.
        A buffer will be allocated for each `data` item.
        Compared with `self.extend`, this method adds a single sample (instead
        of a "batch") to the local buffers.

        Args:
            data: each item will be converted into a torch tensor.
                they will be stacked at the 0-th dim with a new dimension when `get_buffer()` is called.

        """
        if self._buffers is None:
            self._buffers = [[] for _ in data]
        for b, d in zip(self._buffers, data):
            # converting to pytorch tensors so that we can use the distributed API
            d_t, *_ = convert_data_type(d, output_type=torch.Tensor, wrap_sequence=True)
            b.append(d_t)
        self._synced = False

    @abstractmethod
    def aggregate(self, *args: Any, **kwargs: Any):
        """
        Aggregate final results based on the gathered buffers.
        This method is expected to use `get_buffer` to gather the local buffer contents.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def _sync(self):
        """
        All gather the buffers across distributed ranks for aggregating.
        Each buffer will be concatenated as a PyTorch Tensor.

        """
        if self._synced or self._buffers is None:
            return
        try:
            self._synced_tensors = [
                evenly_divisible_all_gather(torch.stack(b, dim=0), concat=True) for b in self._buffers
            ]
        except (RuntimeError, TypeError, ValueError) as e:
            raise TypeError(f"{e}. unable to sync buffer contents: {self._buffers}.") from e
        self._synced = True

    def __len__(self):
        """
        Return the length of the largest buffer.
        Note that the method will trigger synchronization of the local buffers.
        """
        self._sync()
        if not self._synced_tensors:
            return 0
        return max(len(x) for x in self._synced_tensors)

    def get_buffer(self):
        """
        Get the synchronized list of buffers.
        A typical usage is to generate the metrics report based on the raw metric details.
        Each buffer is a PyTorch Tensor.

        """
        self._sync()
        if self._synced_tensors is None:
            return self._synced_tensors
        buffers = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in self._synced_tensors]
        return buffers[0] if len(buffers) == 1 else buffers


class CumulativeIterationMetric(Cumulative, IterationMetric):
    """
    Base class of cumulative metric which collects metrics on each mini-batch data at the iteration level.

    Typically, it computes some intermediate results for each iteration, adds them to the buffers,
    then the buffer contents could be gathered and aggregated for the final result when epoch completed.

    For example, `MeanDice` inherits this class and the usage is as follows:

    .. code-block:: python

        dice_metric = DiceMetric(include_background=True, reduction="mean")

        for val_data in val_loader:
            val_outputs = model(val_data["img"])
            val_outputs = [postprocessing_transform(i) for i in decollate_batch(val_outputs)]
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_data["seg"])  # callable to add metric to the buffer

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()

        # reset the status for next computation round
        dice_metric.reset()

    And to load `predictions` and `labels` from files, then compute metrics with multi-processing, please refer to:
    https://github.com/Project-MONAI/tutorials/blob/master/modules/compute_metric.py.

    """

    def __call__(self, y_pred: TensorOrList, y: Optional[TensorOrList] = None):
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

        Returns:
            The computed metric values at the iteration level.
        """
        ret = super().__call__(y_pred=y_pred, y=y)
        if isinstance(ret, (tuple, list)):
            self.extend(*ret)
        else:
            self.extend(ret)

        return ret

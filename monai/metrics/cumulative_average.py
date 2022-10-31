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

import warnings
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
from numpy.typing import NDArray


class CumulativeAverage:
    """
    A utility class to keep track of average values. For example during training/validation loop,
    we need to accumulate the per-batch metrics and calculate the final average value for the whole dataset.
    When training in multi-gpu environment, with DistributedDataParallel, it will average across the processes.

    Example:

    .. code-block:: python

        from monai.metrics import CumulativeAverage

        run_avg = CumulativeAverage()
        batch_size = 8
        for i in range(len(train_set)):
            ...
            val = calc_metric(x,y) #some metric value
            run_avg.append(val, count=batch_size)

        val_avg = run_avg.aggregate() #average value

    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Reset all  stats
        """
        self.val: torch.Tensor = None  # type: ignore
        self.sum = torch.tensor(0, dtype=torch.float)
        self.count = torch.tensor(0, dtype=torch.float)
        self.is_distributed = dist.is_available() and dist.is_initialized()

    def get_current(self, to_numpy: bool = True) -> Union[NDArray, torch.Tensor]:
        """
        returns the most recent value (averaged across processes)

        Args:
            to_numpy: whether to convert to numpy array. Defaults to True
        """
        if self.val is None:
            return 0

        val = self.val.clone()
        val[~torch.isfinite(val)] = 0

        if self.is_distributed:
            val = val / dist.get_world_size()
            dist.all_reduce(val)

        if to_numpy:
            val = val.cpu().numpy()

        return val

    def aggregate(self, to_numpy: bool = True) -> Union[NDArray, torch.Tensor]:
        """
        returns the total average value (averaged across processes)

        Args:
            to_numpy: whether to convert to numpy array. Defaults to True
        """
        if self.val is None:
            return 0

        sum = self.sum
        count = self.count

        if self.is_distributed:
            sum = sum.to(self.val, copy=True)
            count = count.to(self.val, copy=True)
            dist.all_reduce(sum)
            dist.all_reduce(count)

        val = torch.where(count > 0, sum / count, sum)

        if to_numpy:
            val = val.cpu().numpy()
        return val

    def append(self, val: Any, count: Optional[Any] = 1) -> None:
        """
        Append with a new value, and an optional count. Any data type is supported that is convertable
            with torch.as_tensor() e.g. number, list, numpy array, or Tensor.

        Args:
            val: value (e.g. number, list, numpy array or Tensor) to keep track of
            count: count (e.g. number, list, numpy array or Tensor), to update the contribution count

        For example:
            # a simple constant tracking
            avg = CumulativeAverage()
            avg.append(0.6)
            avg.append(0.8)
            print(avg.aggregate()) #prints 0.7

            # an array tracking, e.g. metrics from 3 classes
            avg= CumulativeAverage()
            avg.append([0.2, 0.4, 0.4])
            avg.append([0.4, 0.6, 0.4])
            print(avg.aggregate()) #prints [0.3, 0.5. 0.4]

            # different contributions / counts
            avg= CumulativeAverage()
            avg.append(1, count=4) #avg metric 1 coming from a batch of 4
            avg.append(2, count=6) #avg metric 2 coming from a batch of 6
            print(avg.aggregate()) #prints 1.6 == (1*4 +2*6)/(4+6)

            # different contributions / counts
            avg= CumulativeAverage()
            avg.append([0.5, 0.5, 0], count=[1, 1, 0]) # last elements count is zero to ignore it
            avg.append([0.5, 0.5, 0.5], count=[1, 1, 1]) #
            print(avg.aggregate()) #prints [0.5, 0.5, 0,5] == ([0.5, 0.5, 0] + [0.5, 0.5, 0.5]) / ([1, 1, 0] + [1, 1, 1])

        """
        self.val = torch.as_tensor(val, dtype=torch.float)
        if self.val.requires_grad:
            self.val = self.val.detach().clone()

        count = torch.as_tensor(count, dtype=torch.float, device="cpu")
        if count.ndim > 0 and count.shape != self.val.shape:
            raise ValueError(
                f"Count shape must match val shape, unless count is a single number: {count} val {self.val.cpu()}"
            )

        val = count * self.val.cpu()

        # account for possible non-finite numbers in val and replace them with 0s
        nfin = torch.isfinite(val)
        if not torch.all(nfin):
            warnings.warn(f"non-finite inputs received: val: {val}, count: {count}")
            count = torch.where(nfin, count, torch.zeros_like(count))
            val = torch.where(nfin, val, torch.zeros_like(val))

        self.count = self.count + count
        self.sum = self.sum + val

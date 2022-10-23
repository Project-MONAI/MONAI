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

from typing import Any, Union

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
        for i in range(len(train_set)):
            ...
            val = calc_metric(x,y) #some metric value
            run_avg.append(val)

        val_avg = run_avg.aggregate() #average value

    """

    def __init__(self, ensure_cpu_group: bool = False) -> None:
        """
        Args:
            ensure_cpu_group: applies only to multi-gpu distributed setup. Optionally request to operate on CPU,
                even if the default DistributedDataParallel group is on GPU. This will create an additional
                CPU group with GLOO backend to reduce across processes on CPU (and maintain all intermediate data on CPU).
                Defaults to False: use the default process group created during DistributedDataParallel initialization.
        """

        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.group = None
        self.device = None

        if self.is_distributed:
            dist_cuda = dist.get_backend() == dist.Backend.NCCL
            if ensure_cpu_group and dist_cuda:
                self.group = dist.new_group(backend="gloo")  # create a new cpu group
            elif dist_cuda:
                self.device = torch.device("cuda", dist.get_rank())

        self.reset()

    def reset(self) -> None:
        """
        Reset all the running status

        """
        self.val: torch.Tensor = 0  # type: ignore
        self.sum: torch.Tensor = 0  # type: ignore
        self.count: torch.Tensor = 0  # type: ignore

    def proper_tensor(self, a: Any) -> torch.Tensor:
        """
        Ensure torch.Tensor float format and optionally copy to the proper gpu device
        """
        x: torch.Tensor = a.detach() if isinstance(a, torch.Tensor) else torch.tensor(a)

        if self.device is not None:
            x = x.to(device=self.device, dtype=torch.float)
        else:
            x = x.cpu().float()

        return x

    def reduce(self, x: torch.Tensor, avg: bool = False) -> torch.Tensor:
        """
        Reduce across processes if in DDP
        """
        if self.is_distributed:
            if avg:
                x = x / dist.get_world_size()
            else:
                x = x.clone()
            dist.all_reduce(x, group=self.group)

        return x

    def get_current(self, np: bool = True) -> Union[NDArray, torch.Tensor]:
        """
        return most recent value (averaged across processes)
        """
        x = self.reduce(self.val, avg=True)
        if np:
            x = x.cpu().numpy()
        return x

    def aggregate(self, np: bool = True) -> Union[NDArray, torch.Tensor]:
        """
        return total average value (averaged across processes)

        """
        sum = self.reduce(self.sum)
        count = self.reduce(self.count)

        x = torch.where(count > 0, sum / count, sum)
        if np:
            x = x.cpu().numpy()
        return x

    def append(self, val: Any, count: Any = 1) -> None:
        """
        Append with a new value, and an optional count

        Args:
            val: new value (e.g. constant, list, numpy array or Tensor) to keep track of.
            count: new count (e.g. constant, list, numpy array or Tensor), to update the contribution count

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

        val = self.proper_tensor(val)
        count = self.proper_tensor(count)

        val_count = val * count
        nfin = ~torch.isfinite(val_count)
        if torch.any(nfin):
            # non-finite numbers may indicate some errors in the user code
            print("non-finite numbers received", val, count)
            zero = torch.tensor(0).to(val)
            val = torch.where(nfin, zero, val)
            count = torch.where(nfin, zero, count)
            val_count = torch.where(nfin, zero, val_count)

        self.val = val
        self.count += count
        self.sum += val_count

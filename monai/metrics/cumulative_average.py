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

import torch

from monai.transforms import isnan
from monai.utils import convert_data_type

from .metric import Cumulative


class CumulativeAverage(Cumulative):
    """
    Cumulatively record data value and aggregate for the average value.
    It supports single class or multi-class data, for example,
    value can be 0.44 (a loss value) or [0.3, 0.4] (metrics of two classes).
    It also supports distributed data parallel, sync data when aggregating.
    For example, recording loss values and compute the overall average value in every 5 iterations:

    .. code-block:: python

        average = CumulativeAverage()
        for i, d in enumerate(dataloader):
            loss = ...
            average.append(loss)
            if i % 5 == 0:
                print(f"cumulative average of loss: {average.aggregate()}")
        average.reset()

    """

    def __init__(self) -> None:
        super().__init__()
        self.sum = None
        self.not_nans = None

    def reset(self):
        """
        Reset all the running status, including buffers, sum, not nans count, etc.

        """
        super().reset()
        self.sum = None
        self.not_nans = None

    def aggregate(self):  # type: ignore
        """
        Sync data from all the ranks and compute the average value with previous sum value.

        """
        data = self.get_buffer()

        # compute SUM across the batch dimension
        nans = isnan(data)
        not_nans = convert_data_type((~nans), dtype=torch.float32)[0].sum(0)
        data[nans] = 0
        f = data.sum(0)

        # clear the buffer for next update
        super().reset()
        self.sum = f if self.sum is None else (self.sum + f)
        self.not_nans = not_nans if self.not_nans is None else (self.not_nans + not_nans)

        return self.sum / self.not_nans

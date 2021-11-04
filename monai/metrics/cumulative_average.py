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

from typing import Optional, Union

import torch

from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction

from .metric import CumulativeIterationMetric


class CumulativeAverage(CumulativeIterationMetric):
    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.sum = None
        self.avg = None
        self.count = 0

    def _compute_tensor(self, value: torch.Tensor, y: Optional[torch.Tensor] = None):
        return value

    def aggregate(self):  # type: ignore
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, self.reduction)
        return (f, not_nans) if self.get_not_nans else f

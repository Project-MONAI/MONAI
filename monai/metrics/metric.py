# Copyright 2020 MONAI Consortium
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


class Metric(ABC):
    """
    Abstract base class for class based API for metrics
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        """
        Compute metric
        """
        raise NotImplementedError


class CumulativeMetric(Metric):
    """
    Abstract base class for metrics which need to be computed over multiple
    samples and need to store intermediate values. To be consistent with
    the other metrics, the metric can be called after all samples
    were added to compute the final result.
    """

    @abstractmethod
    def add_sample(self, *args, **kwargs) -> None:
        """
        Add a new sample for evaluation
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """
        Reset internally saved values
        """
        raise NotImplementedError

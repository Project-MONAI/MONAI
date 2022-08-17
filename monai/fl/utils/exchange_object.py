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

from typing import Dict, Optional

from monai.fl.utils.constants import WeightType


class ExchangeObject(dict):
    """
    Contains the information shared between client and server.

    Args:
        weights: model weights.
        optim: optimizer weights.
        metrics: evaluation metrics.
        weight_type: type of weights (see monai.fl.utils.constants.WeightType).
        statistics: training statistics, i.e. number executed iterations.
    """

    def __init__(
        self,
        weights=None,
        optim=None,
        metrics: Optional[Dict] = None,
        weight_type: Optional[Dict] = None,
        statistics: Optional[Dict] = None,
    ):
        super().__init__()
        self.weights = weights
        self.optim = optim
        self.metrics = metrics
        self.weight_type = weight_type
        self.statistics = statistics
        self._summary = {}

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        if metrics is not None:
            if not isinstance(metrics, dict):
                raise ValueError(f"Expected metrics to be of type dict but received {type(metrics)}")
        self._metrics = metrics

    @property
    def statistics(self):
        return self._statistics

    @statistics.setter
    def statistics(self, statistics):
        if statistics is not None:
            if not isinstance(statistics, dict):
                raise ValueError(f"Expected statistics to be of type dict but received {type(statistics)}")
        self._statistics = statistics

    @property
    def weight_type(self):
        return self._weight_type

    @weight_type.setter
    def weight_type(self, weight_type):
        if weight_type is not None:
            if weight_type not in [WeightType.WEIGHTS, WeightType.WEIGHT_DIFF]:
                raise ValueError(f"Expected weight type to be either {WeightType.WEIGHTS} or {WeightType.WEIGHT_DIFF}")
        self._weight_type = weight_type

    def is_valid_weights(self):
        if not self.weights:
            return False
        if not self.weight_type:
            return False
        return True

    def _add_to_summary(self, key, value):
        if value:
            if isinstance(value, dict):
                self._summary[key] = len(value)
            elif isinstance(value, WeightType):
                self._summary[key] = value
            else:
                self._summary[key] = type(value)

    def summary(self):
        self._summary.update(self)
        for k, v in zip(
            ["weights", "optim", "metrics", "weight_type", "statistics"],
            [self.weights, self.optim, self.metrics, self.weight_type, self.statistics],
        ):
            self._add_to_summary(k, v)
        return self._summary

    def __repr__(self):
        return str(self.summary())

    def __str__(self):
        return str(self.summary())

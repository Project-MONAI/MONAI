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
    """Exchange object

    Contains the information shared between client and server

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
            if weight_type not in [WeightType.WEIGHTS, WeightType.WEIGHT_DIFF]:  # TODO: get all WeightType options?
                raise ValueError(f"Expected weight type to be either {WeightType.WEIGHTS} or {WeightType.WEIGHT_DIFF}")
        self._weight_type = weight_type

    # TODO: add self validation functions?
    def is_valid_weights(self):
        if not self.weights:
            # raise ValueError(f"ExchangeObject doesn't contain a model")
            return False
        if not self.weight_type:
            # raise ValueError(f"ExchangeObject doesn't set weight type")
            return False
        return True

    # TODO: implement better summary function?
    def summary(self):
        return dir(self)

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

from collections import UserDict
from functools import partial
from monai.transforms.utils_pytorch_numpy_unification import max, mean, median, min, percentile, std
from typing import Any

class Operations(UserDict): 
    def evaluate(self, data: Any, **kwargs) -> dict:
        return {k: v(data, **kwargs) for k, v in self.data.items() if callable(v)}

class SampleOperations(Operations):
    # todo: missing value/nan/inf
    def __init__(self) -> None:
        self.data = {
            "max": max,
            "mean": mean,
            "median": median,
            "min": min,
            "stdev": std,
            "percentile": partial(percentile, q=[0.5, 10, 90, 99.5])
        }
        self.data_addon = {
            "percentile_00_5": ("percentile", 0),
            "percentile_10_0": ("percentile", 1),
            "percentile_90_0": ("percentile", 2),
            "percentile_99_5": ("percentile", 3),
        }
    
    def evaluate(self, data: Any, **kwargs) -> dict:
        ret = super().evaluate(data, **kwargs)
        for k, v in self.data_addon.items():
            cache = v[0]
            idx = v[1]
            if isinstance(v, tuple) and cache in ret:
                ret.update({k: ret[cache][idx]})

        return ret

class SummaryOperations(Operations):
    def __init__(self) -> None:
        self.data = {
                "max": max,
                "mean": mean,
                "median": mean,
                "min": min,
                "stdev": mean,
                "percentile_00_5": mean,
                "percentile_10_0": mean,
                "percentile_90_0": mean,
                "percentile_99_5": mean,
            }
    
    def evaluate(self, data: Any, **kwargs) -> dict:
        return {k: v(data[k], **kwargs) for k, v in self.data.items() if callable(v)} 

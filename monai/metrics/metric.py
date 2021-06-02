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

from typing import Any, Optional
from abc import ABC, abstractmethod

import torch

from monai.config import TensorList


class Metric(ABC):
    def __call__(self, y_pred: TensorList, y: Optional[TensorList] = None):
        if isinstance(y_pred, (list, tuple)) or isinstance(y, (list, tuple)):
            # if y_pred or y is a list of channel-first data, add batch dim and compute metric
            if y is not None:
                ret = [self._apply(p_.unsqueeze(0), y_.unsqueeze(0)) for p_, y_ in zip(y_pred, y)]
            else:
                ret = [self._apply(p_.unsqueeze(0), None) for p_ in y_pred]
        else:
            ret = self._apply(y_pred, y)
        return ret

    @abstractmethod
    def _apply(self, y_pred: torch.Tensor, y: Optional[torch.Tensor] = None):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def reduce(self, data: Any):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

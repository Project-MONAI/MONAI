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
from typing import Any

__all__ = ["Bundle"]


class Bundle(ABC):
    """
    Base class for the bundle specification.
    """

    @abstractmethod
    def initialize(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def finalize(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

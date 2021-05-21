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

from abc import ABC
from typing import Callable, Dict


class Handler(ABC):
    """
    Base class of all handlers

    """
    def __init__(self) -> None:
        self.event_funcs = {}

    def get_event_funcs(self):
        return self.event_funcs

    def _register(self, event: str, func: Callable[[Dict], None]):
        """
        Register a function to specified event, the function must take a `dict` arg: "data".
        For example: ``def iteration_completed(self, data: Dict) -> None``.

        """
        self.event_funcs[event] = func

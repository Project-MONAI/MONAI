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
from typing import Dict, Callable
from monai.utils import Events


class Handler(ABC):
    """
    Base class of all handlers

    """
    def __init__(self) -> None:
        self.event_funcs = {}

    def get_event_funcs(self):
        return self.event_funcs

    def _register(self, event: str, func: Callable):
        self.event_funcs[event] = func


class TestHandler(Handler):
    def __init__(self) -> None:
        super().__init__()
        self._register(event=Events.STARTED, func=self._started)
        self._register(event=Events.ITERATION_COMPLETED, func=self._iteration)
        self._register(event=Events.EPOCH_COMPLETED, func=self._epoch)

    def _started(self, data: Dict):
        print(f"total epochs: {data['state']['max_epochs']}.")
        data["state"]["magic_number"] = 123

    def _iteration(self, data: Dict):
        print(f"current iteration: {data['state']['iteration']}.")
        print(f"magic number: {data['state']['magic_number']}")
        data["state"]["magic_number"] += 1

    def _epoch(self, data: Dict):
        print(f"should terminated: {data['should_terminate']}.")
        print(f"end magic number: {data['state']['magic_number']}")

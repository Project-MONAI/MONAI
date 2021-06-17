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

from typing import TYPE_CHECKING, Callable

from monai.engines.utils import IterationEvents, engine_apply_transform
from monai.utils import exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")


class PostProcessing:
    """
    Ignite handler to execute additional post processing after the post transforms in engines.
    So users can insert other handlers between post transforms and this post processing handler.

    """

    def __init__(self, transform: Callable) -> None:
        """
        Args:
            transform: callable function to execute on the `engine.state.batch` and `engine.state.output`.
                can also be composed post transforms.

        """
        self.transform = transform

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(IterationEvents.MODEL_COMPLETED, self)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.state.batch, engine.state.output = engine_apply_transform(
            batch=engine.state.batch,
            output=engine.state.output,
            transform=self.transform,
        )

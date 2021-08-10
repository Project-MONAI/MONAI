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

from monai.config import IgniteInfo
from monai.engines.utils import IterationEvents, engine_apply_transform
from monai.utils import min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


class PostProcessing:
    """
    Ignite handler to execute additional post processing after the post processing in engines.
    So users can insert other handlers between engine postprocessing and this post processing handler.
    If using components from `monai.transforms` as the `transform`, recommend to decollate `engine.state.batch`
    and `engine.state.batch` in the engine(set `decollate=True`) or in the `DecollateBatch` handler first.

    """

    def __init__(self, transform: Callable, event: str = "MODEL_COMPLETED") -> None:
        """
        Args:
            transform: callable function to execute on the `engine.state.batch` and `engine.state.output`.
                can also be composed transforms.
            event: expected EVENT to attach the handler, should be "MODEL_COMPLETED" or "ITERATION_COMPLETED".
                default to "MODEL_COMPLETED".

        """
        self.transform = transform
        event = event.upper()
        if event not in ("MODEL_COMPLETED", "ITERATION_COMPLETED"):
            raise ValueError("event should be `MODEL_COMPLETED` or `ITERATION_COMPLETED`.")
        self.event = event

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.event == "MODEL_COMPLETED":
            engine.add_event_handler(IterationEvents.MODEL_COMPLETED, self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if not isinstance(engine.state.batch, list) or not isinstance(engine.state.output, list):
            engine.state.batch, engine.state.output = engine_apply_transform(
                batch=engine.state.batch,
                output=engine.state.output,
                transform=self.transform,
            )
        else:
            for i, (b, o) in enumerate(zip(engine.state.batch, engine.state.output)):
                engine.state.batch[i], engine.state.output[i] = engine_apply_transform(b, o, self.transform)

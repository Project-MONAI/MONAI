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

from typing import TYPE_CHECKING, Optional

from monai.config import IgniteInfo, KeysCollection
from monai.engines.utils import IterationEvents
from monai.transforms import Decollated
from monai.utils import min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


class DecollateBatch:
    """
    Ignite handler to execute the `decollate batch` logic for `engine.state.batch` and `engine.state.output`.
    Typical usage is to set `decollate=False` in the engine and execute some postprocessing logic first
    then decollate the batch, otherwise, engine will decollate batch before the postprocessing.

    Args:
        event: expected EVENT to attach the handler, should be "MODEL_COMPLETED" or "ITERATION_COMPLETED".
            default to "MODEL_COMPLETED".
        detach: whether to detach the tensors. scalars tensors will be detached into number types
            instead of torch tensors.
        decollate_batch: whether to decollate `engine.state.batch` of ignite engine.
        batch_keys: if `decollate_batch=True`, specify the keys of the corresponding items to decollate
            in `engine.state.batch`, note that it will delete other keys not specified. if None,
            will decollate all the keys. it replicates the scalar values to every item of the decollated list.
        decollate_output: whether to decollate `engine.state.output` of ignite engine.
        output_keys: if `decollate_output=True`, specify the keys of the corresponding items to decollate
            in `engine.state.output`, note that it will delete other keys not specified. if None,
            will decollate all the keys. it replicates the scalar values to every item of the decollated list.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        event: str = "MODEL_COMPLETED",
        detach: bool = True,
        decollate_batch: bool = True,
        batch_keys: Optional[KeysCollection] = None,
        decollate_output: bool = True,
        output_keys: Optional[KeysCollection] = None,
        allow_missing_keys: bool = False,
    ):
        event = event.upper()
        if event not in ("MODEL_COMPLETED", "ITERATION_COMPLETED"):
            raise ValueError("event should be `MODEL_COMPLETED` or `ITERATION_COMPLETED`.")
        self.event = event

        self.batch_transform = (
            Decollated(keys=batch_keys, detach=detach, allow_missing_keys=allow_missing_keys)
            if decollate_batch
            else None
        )

        self.output_transform = (
            Decollated(keys=output_keys, detach=detach, allow_missing_keys=allow_missing_keys)
            if decollate_output
            else None
        )

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
        if self.batch_transform is not None:
            engine.state.batch = self.batch_transform(engine.state.batch)
        if self.output_transform is not None:
            engine.state.output = self.output_transform(engine.state.output)

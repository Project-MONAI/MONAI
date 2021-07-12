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

import warnings
from typing import TYPE_CHECKING, Callable

from monai.config import IgniteInfo
from monai.data import decollate_batch, rep_scalar_to_batch
from monai.engines.utils import IterationEvents, engine_apply_transform
from monai.utils import min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


class DecollateBatch:
    """
    Ignite handler to execute the `decollate batch` logic for `engine.state.batch` and `engine.state.output`.
    So users can set `decollate=False` in the engine and execute some postprocessing logic first
    then decollate the batch, otherwise, engine will decollate batch before the postprocessing.

    """

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
        # replicate the scalar values to make sure all the items have batch dimension, then decollate
        engine.state.batch = decollate_batch(rep_scalar_to_batch(engine.state.batch), detach=True)
        engine.state.output = decollate_batch(rep_scalar_to_batch(engine.state.output), detach=True)

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

from __future__ import annotations

from typing import TYPE_CHECKING

from monai.engines.evaluator import Evaluator
from monai.utils import IgniteInfo, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


class ValidationHandler:
    """
    Attach validator to the trainer engine in Ignite.
    It can support to execute validation every N epochs or every N iterations.

    """

    def __init__(
        self, interval: int, validator: Evaluator | None = None, epoch_level: bool = True, exec_at_start: bool = False
    ) -> None:
        """
        Args:
            interval: do validation every N epochs or every N iterations during training.
            validator: run the validator when trigger validation, suppose to be Evaluator.
                if None, should call `set_validator()` before training.
            epoch_level: execute validation every N epochs or N iterations.
                `True` is epoch level, `False` is iteration level.
            exec_at_start: whether to execute a validation first when starting the training.
                default to `False`. It can be useful especially for some transfer-learning cases
                to validate the initial model before training.

        Raises:
            TypeError: When ``validator`` is not a ``monai.engines.evaluator.Evaluator``.

        """
        if validator is not None and not isinstance(validator, Evaluator):
            raise TypeError(f"validator must be a monai.engines.evaluator.Evaluator but is {type(validator).__name__}.")
        self.validator = validator
        self.interval = interval
        self.epoch_level = epoch_level
        self.exec_at_start = exec_at_start

    def set_validator(self, validator: Evaluator) -> None:
        """
        Set validator if not setting in the __init__().
        """
        if not isinstance(validator, Evaluator):
            raise TypeError(f"validator must be a monai.engines.evaluator.Evaluator but is {type(validator).__name__}.")
        self.validator = validator

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self)
        if self.exec_at_start:
            engine.add_event_handler(Events.STARTED, self)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.validator is None:
            raise RuntimeError("please set validator in __init__() or call `set_validator()` before training.")
        self.validator.run(engine.state.epoch)

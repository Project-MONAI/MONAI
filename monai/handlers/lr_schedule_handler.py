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

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from monai.utils import IgniteInfo, ensure_tuple, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine", as_type="decorator"
    )


class LrScheduleHandler:
    """
    Ignite handler to update the Learning Rate based on PyTorch LR scheduler.
    """

    def __init__(
        self,
        lr_scheduler: _LRScheduler | ReduceLROnPlateau,
        print_lr: bool = True,
        name: str | None = None,
        epoch_level: bool = True,
        step_transform: Callable[[Engine], Any] = lambda engine: (),
    ) -> None:
        """
        Args:
            lr_scheduler: typically, lr_scheduler should be PyTorch
                lr_scheduler object. If customized version, must have `step` and `get_last_lr` methods.
            print_lr: whether to print out the latest learning rate with logging.
            name: identifier of logging.logger to use, if None, defaulting to ``engine.logger``.
            epoch_level: execute lr_scheduler.step() after every epoch or every iteration.
                `True` is epoch level, `False` is iteration level.
            step_transform: a callable that is used to transform the information from `engine`
                to expected input data of lr_scheduler.step() function if necessary.

        Raises:
            TypeError: When ``step_transform`` is not ``callable``.

        """
        self.lr_scheduler = lr_scheduler
        self.print_lr = print_lr
        self.logger = logging.getLogger(name)
        self.epoch_level = epoch_level
        if not callable(step_transform):
            raise TypeError(f"step_transform must be callable but is {type(step_transform).__name__}.")
        self.step_transform = step_transform

        self._name = name

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self._name is None:
            self.logger = engine.logger
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED, self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        args = ensure_tuple(self.step_transform(engine))
        self.lr_scheduler.step(*args)
        if self.print_lr:
            self.logger.info(f"Current learning rate: {self.lr_scheduler._last_lr[0]}")  # type: ignore[union-attr]

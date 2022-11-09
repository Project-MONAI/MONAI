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

from typing import TYPE_CHECKING, Callable, Optional

from monai.config import IgniteInfo
from monai.utils import min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
EarlyStopping, _ = optional_import("ignite.handlers", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EarlyStopping")

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine", as_type="decorator"
    )


class EarlyStopHandler:
    """
    EarlyStopHandler acts as an Ignite handler to stop training if no improvement after a given number of events.
    It‘s based on the `EarlyStopping` handler in ignite.

    Args:
        patience: number of events to wait if no improvement and then stop the training.
        score_function: It should be a function taking a single argument, an :class:`~ignite.engine.engine.Engine`
            object that the handler attached, can be a trainer or validator, and return a score `float`.
            an improvement is considered if the score is higher.
        trainer: trainer engine to stop the run if no improvement, if None, must call `set_trainer()` before training.
        min_delta: a minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count as no improvement.
        cumulative_delta: if True, `min_delta` defines an increase since the last `patience` reset, otherwise,
            it defines an increase after the last event, default to False.
        epoch_level: check early stopping for every epoch or every iteration of the attached engine,
            `True` is epoch level, `False` is iteration level, default to epoch level.

    Note:
        If in distributed training and uses loss value of every iteration to detect early stopping,
        the values may be different in different ranks.
        User may attach this handler to validator engine to detect validation metrics and stop the training,
        in this case, the `score_function` is executed on validator engine and `trainer` is the trainer engine.

    """

    def __init__(
        self,
        patience: int,
        score_function: Callable,
        trainer: Optional[Engine] = None,
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
        epoch_level: bool = True,
    ) -> None:
        self.patience = patience
        self.score_function = score_function
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.epoch_level = epoch_level
        self._handler = None

        if trainer is not None:
            self.set_trainer(trainer=trainer)

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED, self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def set_trainer(self, trainer: Engine):
        """
        Set trainer to execute early stop if not setting properly in `__init__()`.
        """
        self._handler = EarlyStopping(
            patience=self.patience,
            score_function=self.score_function,
            trainer=trainer,
            min_delta=self.min_delta,
            cumulative_delta=self.cumulative_delta,
        )

    def __call__(self, engine: Engine) -> None:
        if self._handler is None:
            raise RuntimeError("please set trainer in __init__() or call set_trainer() before training.")
        self._handler(engine)

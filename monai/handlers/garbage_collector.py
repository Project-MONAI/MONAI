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

import gc
from typing import TYPE_CHECKING

from monai.config import IgniteInfo
from monai.utils import min_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")


class GarbageCollector:
    """
    Run garbage collector after each epoch

    Args:
        trigger_event: the event that trigger a call to this handler.
            - "epoch", after completion of each epoch (equivalent of ignite.engine.Events.EPOCH_COMPLETED)
            - "iteration", after completion of each iteration (equivalent of ignite.engine.Events.ITERATION_COMPLETED)
            - any ignite built-in event from ignite.engine.Events.
            Defaults to "epoch".
        log_level: log level (integer) for some garbage collection information as below. Defaults to 10 (DEBUG).
            - 50 (CRITICAL)
            - 40 (ERROR)
            - 30 (WARNING)
            - 20 (INFO)
            - 10 (DEBUG)
            - 0 (NOTSET)
    """

    def __init__(self, trigger_event: str = "epoch", log_level: int = 10):
        if isinstance(trigger_event, Events):
            self.trigger_event = trigger_event
        elif trigger_event.lower() == "epoch":
            self.trigger_event = Events.EPOCH_COMPLETED
        elif trigger_event.lower() == "iteration":
            self.trigger_event = Events.ITERATION_COMPLETED
        else:
            raise ValueError(
                f"'trigger_event' should be either epoch, iteration, or an ignite built-in event from"
                f" ignite.engine.Events, '{trigger_event}' was given."
            )

        self.log_level = log_level

    def attach(self, engine: Engine) -> None:
        if not engine.has_event_handler(self, self.trigger_event):
            engine.add_event_handler(self.trigger_event, self)

    def __call__(self, engine: Engine) -> None:
        """
        This method calls python garbage collector.

        Args:
            engine: Ignite Engine, it should be either a trainer or validator.
        """
        # get count before garbage collection
        pre_count = gc.get_count()
        # fits call to garbage collector
        gc.collect()
        # second call to garbage collector
        unreachable = gc.collect()
        # get count after garbage collection
        after_count = gc.get_count()
        engine.logger.log(
            self.log_level,
            f"Garbage Count: [before: {pre_count}] -> [after: {after_count}] (unreachable : {unreachable})",
        )

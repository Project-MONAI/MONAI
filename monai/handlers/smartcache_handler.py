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

from typing import TYPE_CHECKING

from monai.data import SmartCacheDataset
from monai.utils import exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")


class SmartCacheHandler:
    """
    Attach SmartCache logic to the engine in Ignite.
    Mainly include the `start`, `update_cache`, and `shutdown` functions of SmartCacheDataset.

    """

    def __init__(self, smartcacher: SmartCacheDataset) -> None:
        """
        Args:
            smartcacher: predefined SmartCacheDataset, will attach it to the engine.

        Raises:
            TypeError: When ``smartcacher`` is not a ``monai.data.SmartCacheDataset``.

        """
        if not isinstance(smartcacher, SmartCacheDataset):
            raise TypeError("smartcacher must be a monai.data.SmartCacheDataset.")
        self.smartcacher = smartcacher

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(Events.STARTED, self.started)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_completed)
        engine.add_event_handler(Events.COMPLETED, self.completed)

    def started(self, engine: Engine) -> None:
        """Callback for train or validation/evaluation started Event.
        Start the replacement thread of SmartCacheDataset.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        self.smartcacher.start()

    def epoch_completed(self, engine: Engine) -> None:
        """Callback for train or validation/evaluation epoch completed Event.
        Update cache content with replacement data.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        self.smartcacher.update_cache()

    def completed(self, engine: Engine) -> None:
        """Callback for train or validation/evaluation completed Event.
        Stop the replacement thread of SmartCacheDataset.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        self.smartcacher.shutdown()

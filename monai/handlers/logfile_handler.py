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

import logging
import os
from typing import TYPE_CHECKING, Optional

from monai.config import IgniteInfo
from monai.utils import min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


__all__ = ["LogfileHandler"]


class LogfileHandler:
    """
    Adds a `logging.FileHandler` to the attached engine's logger when the start event occurs and removes it again when
    then completed event occurs.

    A handler is needed to remove `FileHandler` object when the complete event occurs so that further runs of different
    engines write only to the log files they should, rather than previous files. Multiple handlers can write to the same
    file which allows output from train and evaluation engine objects to be condensed in one file. If the given output
    directory doesn't exist it will by default be created when the  start event occurs. This can be used in conjunction
    with `CheckpointSaver` to save a log file to the same destination as the saved checkpoints. Since the handler is
    added possibly after other logging events during initialisation, not all logging data will be retained.

    Args:
        output_dir: directory to save the log file to
        filename: name of the file to save log to
        loglevel: log level for the handler
        formatter: format string for the `logging.Formatter` set for the handler
        create_dir: if True, create `output_dir` if it doesn't exist
    """

    def __init__(
        self,
        output_dir: str,
        filename: str = "log.txt",
        loglevel: int = logging.INFO,
        formatter: str = "%(asctime)s %(name)s %(levelname)s: %(message)s",
        create_dir: bool = True,
    ):
        self.output_dir: str = output_dir
        self.filename: str = filename
        self.loglevel: int = loglevel
        self.formatter: str = formatter
        self.create_dir: bool = create_dir
        self.logger: Optional[logging.Logger] = None
        self.handler: Optional[logging.FileHandler] = None

    def attach(self, engine: Engine) -> None:
        self.logger = engine.logger
        engine.add_event_handler(Events.STARTED, self._start)
        engine.add_event_handler(Events.COMPLETED, self._completed)

    def _start(self, engine: Engine) -> None:
        if self.create_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.handler = logging.FileHandler(os.path.join(self.output_dir, self.filename))
        self.handler.setLevel(self.loglevel)
        self.handler.setFormatter(logging.Formatter(self.formatter))
        
        if self.logger is not None:
            self.logger.addHandler(self.handler)
        else:
            raise AttributeError("`self.logger` should not be None in start event")

    def _completed(self, engine: Engine) -> None:
        if self.logger is not None:
            self.logger.removeHandler(self.handler)
        else:
            raise AttributeError("`self.logger` should not be None in complete event")
            
        if self.handler is not None:
            self.handler.close()
        else:
            raise AttributeError("`self.handler` should not be None in complete event")
            
        self.handler = None

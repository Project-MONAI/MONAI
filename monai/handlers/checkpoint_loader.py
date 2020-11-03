# Copyright 2020 MONAI Consortium
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
from typing import TYPE_CHECKING, Dict, Optional

import os
import torch

from monai.utils import exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Events")
Checkpoint, _ = optional_import("ignite.handlers", "0.4.2", exact_version, "Checkpoint")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")


class CheckpointLoader:
    """
    CheckpointLoader acts as an Ignite handler to load checkpoint data from file.
    It can load variables for network, optimizer, lr_scheduler, etc.
    If saving checkpoint after `torch.nn.DataParallel`, need to save `model.module` instead
    as PyTorch recommended and then use this loader to load the model.

    Args:
        load_path: the file path of checkpoint, it should be a PyTorch `pth` file.
        load_dict: target objects that load checkpoint to. examples::

            {'network': net, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

        name: identifier of logging.logger to use, if None, defaulting to ``engine.logger``.
        map_location: when loading the module for distributed training/evaluation,
            need to provide an appropriate map_location argument to prevent a process
            to step into others’ devices. If map_location is missing, torch.load will
            first load the module to CPU and then copy each parameter to where it was
            saved, which would result in all processes on the same machine using the
            same set of devices.

    """

    def __init__(
        self,
        load_path: str,
        load_dict: Dict,
        name: Optional[str] = None,
        map_location: Optional[Dict] = None,
        find_latest: bool = False,
    ) -> None:
        assert load_path is not None, "must provide clear path to load checkpoint."
        self.load_path = load_path
        assert (find_latest and self.load_path[-3:] != '.pt') or \
               self.load_path[-3:] == '.pt' and not find_latest, "either provide an exact path, or set find latest to true"
        assert load_dict is not None and len(load_dict) > 0, "must provide target objects to load."
        self.logger = logging.getLogger(name)
        for k, v in load_dict.items():
            if hasattr(v, "module"):
                load_dict[k] = v.module
        self.load_dict = load_dict
        self._name = name
        self.map_location = map_location
        self.find_latest = find_latest

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self._name is None:
            self.logger = engine.logger
        engine.add_event_handler(Events.STARTED, self)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.find_latest:
            models = sorted(os.listdir(self.load_path))
            self.logger.info(
                "This directory already exisits, looking for previous models.")
            if len(models) > 1:
                list_model_time = []
                times = []
                for model_ in models:
                    loc = os.path.join(self.load_path, model_)
                    edit_time = os.path.getmtime(loc)
                    times.append(edit_time)
                    list_model_time.append([edit_time, loc])
                list_model_time.sort(key=lambda x: x[0])
                checkpoint = torch.load(list_model_time[-1][1], map_location=self.map_location)
                if len(self.load_dict) == 1:
                    key = list(self.load_dict.keys())[0]
                    if not (key in checkpoint):
                        checkpoint = {key: checkpoint}
                Checkpoint.load_objects(to_load=self.load_dict,
                                        checkpoint=checkpoint)
                self.logger.info(
                    f"Restored all variables from {list_model_time[-1][1]}")
            else:
                self.logger.info("No models found, resuming normally.")
        else:
            checkpoint = torch.load(self.load_path, map_location=self.map_location)
            if len(self.load_dict) == 1:
                key = list(self.load_dict.keys())[0]
                if not (key in checkpoint):
                    checkpoint = {key: checkpoint}

            Checkpoint.load_objects(to_load=self.load_dict, checkpoint=checkpoint)
            self.logger.info(f"Restored all variables from {self.load_path}")

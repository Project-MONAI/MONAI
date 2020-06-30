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
from typing import Optional

import torch

from monai.utils import exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Events")
Engine, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Engine")
Checkpoint, _ = optional_import("ignite.handlers", "0.3.0", exact_version, "Checkpoint")


class CheckpointLoader:
    """
    CheckpointLoader acts as an Ignite handler to load checkpoint data from file.
    It can load variables for network, optimizer, lr_scheduler, etc.
    If saving checkpoint after `torch.nn.DataParallel`, need to save `model.module` instead
    as PyTorch recommended and then use this loader to load the model.

    Args:
        load_path: the file path of checkpoint, it should be a PyTorch `pth` file.
        load_dict (dict): target objects that load checkpoint to. examples::

            {'network': net, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

        name: identifier of logging.logger to use, if None, defaulting to ``engine.logger``.

    """

    def __init__(self, load_path: str, load_dict, name: Optional[str] = None):
        assert load_path is not None, "must provide clear path to load checkpoint."
        self.load_path = load_path
        assert load_dict is not None and len(load_dict) > 0, "must provide target objects to load."
        self.logger = logging.getLogger(name)
        for k, v in load_dict.items():
            if hasattr(v, "module"):
                load_dict[k] = v.module
        self.load_dict = load_dict

        self._name = name

    def attach(self, engine: Engine):
        if self._name is None:
            self.logger = engine.logger
        return engine.add_event_handler(Events.STARTED, self)

    def __call__(self, engine):
        checkpoint = torch.load(self.load_path)
        if len(self.load_dict) == 1:
            key = list(self.load_dict.keys())[0]
            if not (key in checkpoint):
                checkpoint = {key: checkpoint}

        Checkpoint.load_objects(to_load=self.load_dict, checkpoint=checkpoint)
        self.logger.info(f"Restored all variables from {self.load_path}")

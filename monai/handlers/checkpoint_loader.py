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
import torch
from ignite.engine import Events
from ignite.handlers import Checkpoint


class CheckpointLoader:
    """
    CheckpointLoader acts as an Ignite handler to load checkpoint data from file.
    It can load variables for network, optimizer, lr_scheduler.
    And also can restore training if load the state_dict of Ignite engine.

    Args:
        load_path (str): the file path of checkpoint, it should be a PyTorch pth file.
        load_dict (dict): target objects that load checkpoint to. examples::

            {'network': net, 'optimizer': optimizer, 'engine', engine}

        name (str): identifier of logging.logger to use, if None, defaulting to ``engine.logger``.

    """

    def __init__(self, load_path, load_dict, name=None):
        assert load_path is not None, "must provide clear path to load checkpoint."
        self.load_path = load_path
        assert load_dict is not None and len(load_dict) > 0, "must provide target objects to load."
        self.logger = None if name is None else logging.getLogger(name)
        for k, v in load_dict.items():
            if hasattr(v, "module"):
                load_dict[k] = v.module
        self.load_dict = load_dict

    def attach(self, engine):
        if self.logger is None:
            self.logger = engine.logger
        return engine.add_event_handler(Events.STARTED, self)

    def __call__(self, engine):
        checkpoint = torch.load(self.load_path)
        Checkpoint.load_objects(to_load=self.load_dict, checkpoint=checkpoint)
        self.logger.info(f"Restored all variables from {self.load_path}")

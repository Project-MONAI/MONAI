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
import warnings
from typing import TYPE_CHECKING

import torch

from monai.networks.utils import copy_model_state
from monai.utils import IgniteInfo, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
Checkpoint, _ = optional_import("ignite.handlers", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Checkpoint")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


class CheckpointLoader:
    """
    CheckpointLoader acts as an Ignite handler to load checkpoint data from file.
    It can load variables for network, optimizer, lr_scheduler, etc.
    If saving checkpoint after `torch.nn.DataParallel`, need to save `model.module` instead
    as PyTorch recommended and then use this loader to load the model.

    Usage example::

        trainer = SupervisedTrainer(...)
        save_dict = {
            "trainer": trainer,
            "net": network,
            "opt": optimizer,
            "lr": lr_scheduler,
        }

        map_location = "cuda:0"
        # checkpoint needs to have same save_dict for this to work
        handler = CheckpointLoader(load_path="/test/checkpoint.pt", load_dict=save_dict, map_location=map_location, strict=True)
        handler(trainer)
        # Trainer now has the same state as stored, including the number of epochs and iterations completed
        # so you can resume an interrupted training at the place where it left

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
        strict: whether to strictly enforce that the keys and data shape in the `state_dict` of every item
            of `load_dict` match the `state_dict` of the corresponding items of checkpoint, default to `True`.
        strict_shape: whether to enforce the data shape of the matched layers in the checkpoint,
            `if `False`, it will skip the layers that have different data shape with checkpoint content,
            and ignore the `strict` arg. this can be useful advanced feature for transfer learning.
            users should totally understand which layers will have different shape. default to `True`.

    Note: if `strict_shape=False`, will only load checkpoint for `torch.nn.Module` and skip other
        items in the `load_dict`. For example, if the shape of some layers in current model can't
        match the checkpoint, the `parameter_group` of current optimizer may also can't match the
        checkpoint, so skip loading checkpoint for optimizer.

        For more details about loading checkpoint, please refer to:
        https://pytorch.org/ignite/v0.4.5/generated/ignite.handlers.checkpoint.Checkpoint.html
        #ignite.handlers.checkpoint.Checkpoint.load_objects.
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict.

    """

    def __init__(
        self,
        load_path: str,
        load_dict: dict,
        name: str | None = None,
        map_location: dict | None = None,
        strict: bool = True,
        strict_shape: bool = True,
    ) -> None:
        if load_path is None:
            raise AssertionError("must provide clear path to load checkpoint.")
        self.load_path = load_path
        if load_dict is None or len(load_dict) <= 0:
            raise AssertionError("must provide target objects to load.")
        self.logger = logging.getLogger(name)
        self.load_dict = load_dict
        self._name = name
        self.map_location = map_location
        if strict and not strict_shape:
            warnings.warn("as `strict_shape` is already False, change `strict` to False.")
            strict = False
        self.strict = strict
        self.strict_shape = strict_shape

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
        checkpoint = torch.load(self.load_path, map_location=self.map_location)

        k, _ = list(self.load_dict.items())[0]
        # single object and checkpoint is directly a state_dict
        if len(self.load_dict) == 1 and k not in checkpoint:
            checkpoint = {k: checkpoint}

        if not self.strict_shape:
            pop_items: list[str] = []
            for k, obj in self.load_dict.items():
                if isinstance(obj, torch.nn.Module):
                    # skip items that don't match key name or data shape
                    checkpoint[k] = copy_model_state(obj, checkpoint, inplace=False)[0]
                else:
                    warnings.warn("`strict_shape` is False, load checkpoint for model, skip others in `load_dict`.")
                    pop_items.append(k)
            for i in pop_items:
                self.load_dict.pop(i)

        # save current max epochs setting in the engine, don't overwrite it if larger than max_epochs in checkpoint
        prior_max_epochs = engine.state.max_epochs
        Checkpoint.load_objects(to_load=self.load_dict, checkpoint=checkpoint, strict=self.strict)
        if prior_max_epochs is not None and engine.state.epoch > prior_max_epochs:
            raise ValueError(
                f"Epoch count ({engine.state.epoch}) in checkpoint is larger than "
                f"the `engine.state.max_epochs` ({prior_max_epochs}) of engine. To further train from checkpoint, "
                "construct trainer with `max_epochs` larger than checkpoint's epoch count. "
                "To use checkpoint for inference, no need to load state_dict for the engine."
            )
        engine.state.max_epochs = prior_max_epochs

        self.logger.info(f"Restored all variables from {self.load_path}")

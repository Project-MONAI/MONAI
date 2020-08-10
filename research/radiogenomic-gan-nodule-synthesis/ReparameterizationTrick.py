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
"""
Reparameterization Trick: Sec 2.4 https://arxiv.org/pdf/1312.6114.pdf
Let z be a continuous random variable, and z ∼ qφ(z|x) be some conditional distribution.
It is then often possible to express the random variable z as a deterministic variable z = gφ(, x),
    where  is an auxiliary variable with independent marginal p(), 
    and gφ(.) is some vector-valued function parameterized by φ.

Must be updated every iteration, and applied before model inference or checkpoint saving.
"""

import logging
from copy import deepcopy

import torch
from ignite.engine import Engine, Events

_backup_params = None
_average_params_g = None
_logger_name = "Reparameterization"


def _copy_out_params(model: torch.nn.Module):
    params = deepcopy([p.data for p in model.parameters()])
    return params


def _load_in_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


class ReparameterizationUpdate:
    """
    Update average parameters after every iteration or epoch. Ignite Engine handler.
    """

    def __init__(self, network, update_interval: int = 1, epoch_only: bool = False) -> None:
        self.update_interval = update_interval
        self.epoch_only = epoch_only
        self.network = network
        self.logger = logging.getLogger(_logger_name)

        global _average_params_g
        if _average_params_g is None:
            _average_params_g = _copy_out_params(network)

    def update_avg_params(self, engine: Engine) -> None:
        global _average_params_g
        for p, avg_p in zip(self.network.parameters(), _average_params_g):
            avg_p.mul_(0.999).add_(0.001, p.data)

    def attach(self, engine: Engine) -> None:
        if self.epoch_only:
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.update_interval), self.update_avg_params)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.update_interval), self.update_avg_params)


class ReparameterizationApply:
    """
    Loads average params into network. Use before model snapshot or evaluation. Ignite Engine handler.
    """

    def __init__(
        self, network: torch.nn.Module, epoch_level: bool = True, save_final: bool = False, save_interval: int = 0
    ) -> None:
        self.network = network
        self.epoch_level = epoch_level
        self.save_final = save_final
        self.save_interval = save_interval
        self.load_in_params = _load_in_params
        self.copy_out_params = _copy_out_params
        self.logger = logging.getLogger(_logger_name)

    def attach(self, engine: Engine) -> None:
        if self.save_final:
            engine.add_event_handler(Events.COMPLETED, self.completed)
            engine.add_event_handler(Events.EXCEPTION_RAISED, self.exception_raised)
        if self.save_interval > 0:
            if self.epoch_level:
                engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.save_interval), self.interval_completed)
            else:
                engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.save_interval), self.interval_completed)

    def completed(self, engine: Engine) -> None:
        global _average_params_g, _backup_params
        _backup_params = self.copy_out_params(self.network)
        self.load_in_params(self.network, _average_params_g)
        self.logger.info("Training completed, loaded average params into network.")

    def exception_raised(self, engine: Engine, e: Exception) -> None:
        global _average_params_g, _backup_params
        _backup_params = self.copy_out_params(self.network)
        self.load_in_params(self.network, _average_params_g)
        self.logger.info("Exception_raised, loaded average params into network.")

    def interval_completed(self, engine: Engine) -> None:
        global _average_params_g, _backup_params
        _backup_params = self.copy_out_params(self.network)
        self.load_in_params(self.network, _average_params_g)
        if self.epoch_level:
            self.logger.info(f"Loaded average params at epoch: {engine.state.epoch}")
        else:
            self.logger.info(f"Loaded average params at iteration: {engine.state.iteration}")


class ReparameterizationRestore:
    """
    Restore network parameters. Use before resuming training, and after loading average params and performing relevant network operations.
    """

    def __init__(self, network: torch.nn.Module, epoch_level: bool = True, save_interval: int = 0) -> None:
        self.network = network
        self.epoch_level = epoch_level
        self.save_interval = save_interval
        self.load_in_params = _load_in_params
        self.logger = logging.getLogger(_logger_name)

    def attach(self, engine: Engine) -> None:
        if self.save_interval > 0:
            if self.epoch_level:
                engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.save_interval), self.interval_completed)
            else:
                engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.save_interval), self.interval_completed)

    def interval_completed(self, engine: Engine) -> None:
        global _backup_params
        self.load_in_params(self.network, _backup_params)
        if self.epoch_level:
            self.logger.info(f"Restored network params at epoch: {engine.state.epoch}")
        else:
            self.logger.info(f"Restored network params at iteration: {engine.state.iteration}")

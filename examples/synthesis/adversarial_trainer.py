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

from typing import Callable, Dict, Optional, Union, Sequence, TYPE_CHECKING

import torch
from torch.optim.optimizer import Optimizer

from monai.transforms import Transform
from monai.data import DataLoader
from monai.engines import Trainer
from monai.engines.utils import default_prepare_batch
from monai.inferers import Inferer, SimpleInferer
from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", "0.3.0", exact_version, "Metric")


class GanKeys:
    """
    A set of common keys for the adversarial training process.
    """

    REALS = "reals"
    FAKES = "fakes"
    PRED = "pred"
    GLOSS = "g_loss"
    DLOSS = "d_loss"

# from monai.engines.utils import GANKeys as Keys
Keys = GanKeys

class AdversarialTrainer(Trainer):
    """
    Standard GAN adversarial training,  inherits from trainer and Workflow.

    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for engine to run, validator and evaluator have only 1 epoch.
        train_data_loader: Ignite engine use data_loader to run, must be torch.DataLoader.
        network: to train with this network.
        optimizer: the optimizer associated to the network.
        loss_function: the loss function associated to the optimizer.
        prepare_batch: function to parse input data for current iteration
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        amp: whether to enable auto-mixed-precision training, reserved.
        post_transform: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_train_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_train_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        train_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.

    """

    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        train_data_loader: DataLoader,
        g_network: torch.nn.Module,
        g_optimizer: Optimizer,
        g_loss_function: Callable,
        d_network: torch.nn.Module,
        d_optimizer: Optimizer,
        d_loss_function: Callable,
        g_inferer: Inferer = SimpleInferer(),
        d_inferer: Inferer = SimpleInferer(),
        d_train_interval: int = 1,
        d_train_steps: int = 5,
        latent_size: int = 64,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        amp: bool = True,
        post_transform: Optional[Transform] = None,
        key_train_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        train_handlers: Optional[Sequence] = None,
    ):
        # set up Ignite engine and environments
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            amp=amp,
            data_loader=train_data_loader,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            handlers=train_handlers,
            post_transform=post_transform,
        )
        self.g_network = g_network
        self.g_optimizer = g_optimizer
        self.g_loss_function = g_loss_function
        self.g_inferer = g_inferer
        self.d_network = d_network
        self.d_optimizer = d_optimizer
        self.d_loss_function = d_loss_function
        self.d_inferer = d_inferer
        self.d_train_interval = d_train_interval
        self.d_train_steps = d_train_steps
        self.step = 0
        self.latent_size = latent_size

    def _iteration(self, engine: Engine, batchdata: Union[Dict, Sequence]) -> Dict[str, torch.Tensor]:
        """
        Callback function for Adversarial Training processing logic of 1 iteration in Ignite Engine.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: must provide batch data for current iteration.

        """
        if batchdata is None:
            raise ValueError("must provide batch data for current iteration.")
        
        # inputs = self.prepare_batch(batchdata)

        # self.g_network.eval()
        # self.d_network.train()

        # MAKE DATA

        real_data = batchdata.to(engine.state.device)
        batch_size = real_data.shape[0]
        fake_latents = torch.randn(batch_size, self.latent_size).to(engine.state.device)
        fake_data = self.g_inferer(fake_latents, self.g_network)

        # TRAIN DISCRIMINATOR

        d_total_loss = 0
        
        if self.step % self.d_train_interval == 0:
            for _ in range(self.d_train_steps):
                self.d_optimizer.zero_grad()
                dloss = self.d_loss_function(fake_data, real_data)
                dloss.backward()
                self.d_optimizer.step()
                d_total_loss += dloss.item()

        # TRAIN GENERATOR

        # self.g_network.train()
        # self.d_network.eval()

        self.g_optimizer.zero_grad()
        g_loss = self.g_loss_function(g_data)
        g_loss.backward()
        self.g_optimizer.step()
        epoch_loss += loss.item()

        self.step += 1

        return {Keys.REALS: real_data, Keys.FAKES: fake_data, Keys.GLOSS: g_loss, Keys.DLOSS: d_total_loss}



# if self.step % self.d_train_interval == 0:
#     d_total_loss = 0
    
#     for _ in range(d_train_steps):  # 5
#         self.d_optimizer.zero_grad()
#         dloss = self.d_loss_function(fake_data, real_data)
#         dloss.backward()
#         self.d_optimizer.step()
#         d_total_loss += dloss.item()
        
#     d_step_loss.append((self.step, d_total_loss / d_train_steps))

# self.step += 1
     
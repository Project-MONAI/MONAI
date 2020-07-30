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

from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from monai.engines.utils import CommonKeys as Keys
from monai.engines.utils import GanKeys
from monai.engines.utils import default_prepare_batch
from monai.engines.workflow import Workflow
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", "0.3.0", exact_version, "Metric")


class Trainer(Workflow):
    """
    Base class for all kinds of trainers, inherits from Workflow.

    """

    def run(self) -> None:
        """
        Execute training based on Ignite Engine.
        If call this function multiple times, it will continuously run from the previous state.

        """
        if self._is_done(self.state):
            self.state.iteration = 0  # to avoid creating new State instance in ignite Engine.run
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None
        super().run()

    def get_train_stats(self) -> Dict[str, float]:
        return {"total_epochs": self.state.max_epochs, "total_iterations": self.state.epoch_length}


class SupervisedTrainer(Trainer):
    """
    Standard supervised training method with image and label, inherits from trainer and Workflow.

    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for trainer to run.
        train_data_loader: Ignite engine use data_loader to run, must be torch.DataLoader.
        network: to train with this network.
        optimizer: the optimizer associated to the network.
        loss_function: the loss function associated to the optimizer.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        post_transform: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_train_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_train_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        train_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        amp: whether to enable auto-mixed-precision training, default is False.

    """

    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        train_data_loader: DataLoader,
        network: torch.nn.Module,
        optimizer: Optimizer,
        loss_function: Callable,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Inferer = SimpleInferer(),
        post_transform: Optional[Transform] = None,
        key_train_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        train_handlers: Optional[Sequence] = None,
        amp: bool = False,
    ) -> None:
        # set up Ignite engine and environments
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            post_transform=post_transform,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            handlers=train_handlers,
            amp=amp,
        )

        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.inferer = inferer

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]):
        """
        Callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.
            - LOSS: loss value computed by loss function.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        inputs, targets = self.prepare_batch(batchdata)
        inputs, targets = inputs.to(engine.state.device), targets.to(engine.state.device)

        self.network.train()
        self.optimizer.zero_grad()
        if self.amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                predictions = self.inferer(inputs, self.network)
                loss = self.loss_function(predictions, targets).mean()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predictions = self.inferer(inputs, self.network)
            loss = self.loss_function(predictions, targets).mean()
            loss.backward()
            self.optimizer.step()

        return {Keys.IMAGE: inputs, Keys.LABEL: targets, Keys.PRED: predictions, Keys.LOSS: loss.item()}


class AdversarialTrainer(Trainer):
    """
    Standard generative adversarial network training, inherits from trainer and Workflow. 

    Based on [Goodfellow 2014] https://arxiv.org/abs/1406.2661

        Training Loop, for each batch of data size m
            1. Generate m fakes from new latent codes.
            2. Update D with these fakes and current batch reals, repeated d_train_steps times.
            3. Generate m fakes from new latent codes.
            4. Update generator with these fakes using discriminator feedback.

    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for engine to run
        train_data_loader: Real data DataLoader to update discriminator
        g_network: generator (G) network architecture
        g_optimizer: G optimizer function
        g_loss_function: G loss function for optimizer
        d_network: discriminator (D) network architecture
        d_optimizer: D optimizer function
        d_loss_function: D loss function for optimizer
        g_inferer: inference method to execute G model forward on latent code
        d_inferer: inference method to execute D model forward
        d_train_steps: number of times to update D with real data minibatch. 
        latent_shape: size of G random input latent code
        prepare_batch: function to parse real data for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
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
        d_train_steps: int = 1,
        latent_shape: int = 64,
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
        self.d_train_steps = d_train_steps
        self.latent_shape = latent_shape

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

        real_data = self.prepare_batch(batchdata)
        real_data = real_data.to(engine.state.device)

        # Generate Fakes
        batch_size = real_data.shape[0]
        fake_latents = torch.randn(batch_size, self.latent_shape).to(engine.state.device)
        fake_data = self.g_inferer(fake_latents, self.g_network)

        # Train Discriminator
        d_total_loss = 0
        for _ in range(self.d_train_steps):
            self.d_optimizer.zero_grad()
            dloss = self.d_loss_function(fake_data, real_data)
            dloss.backward()
            self.d_optimizer.step()
            d_total_loss += dloss.item()

        # Train Generator
        fake_latents = torch.randn(batch_size, self.latent_shape).to(engine.state.device)
        fake_data = self.g_inferer(fake_latents, self.g_network)
        self.g_optimizer.zero_grad()
        g_loss = self.g_loss_function(fake_data)
        g_loss.backward()
        self.g_optimizer.step()
        g_loss = g_loss.item()

        return {GanKeys.GLOSS: g_loss, GanKeys.DLOSS: d_total_loss}
        # return {Keys.REALS: real_data, Keys.FAKES: fake_data, Keys.GLOSS: g_loss, Keys.DLOSS: d_total_loss}

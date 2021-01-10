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

from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from monai.engines.utils import CommonKeys as Keys
from monai.engines.utils import GanKeys, default_make_latent, default_prepare_batch
from monai.engines.workflow import Workflow
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "Metric")

__all__ = ["Trainer", "SupervisedTrainer", "GanTrainer"]


class Trainer(Workflow):
    """
    Base class for all kinds of trainers, inherits from Workflow.

    """

    def run(self) -> None:
        """
        Execute training based on Ignite Engine.
        If call this function multiple times, it will continuously run from the previous state.

        """
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None
        super().run()

    def get_train_stats(self) -> Dict[str, float]:
        return {"total_epochs": self.state.max_epochs, "total_iterations": self.state.epoch_length}


class SupervisedTrainer(Trainer):
    """
    Standard supervised training method with image and label, inherits from ``Trainer`` and ``Workflow``.

    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for trainer to run.
        train_data_loader: Ignite engine use data_loader to run, must be torch.DataLoader.
        network: to train with this network.
        optimizer: the optimizer associated to the network.
        loss_function: the loss function associated to the optimizer.
        epoch_length: number of iterations for one epoch, default to `len(train_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
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
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
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
            epoch_length=epoch_length,
            non_blocking=non_blocking,
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
        self.inferer = SimpleInferer() if inferer is None else inferer

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
        batch = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        if len(batch) == 2:
            inputs, targets = batch
            args: Tuple = ()
            kwargs: Dict = {}
        else:
            inputs, targets, args, kwargs = batch

        self.network.train()
        self.optimizer.zero_grad()
        if self.amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                predictions = self.inferer(inputs, self.network, *args, **kwargs)
                loss = self.loss_function(predictions, targets).mean()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predictions = self.inferer(inputs, self.network, *args, **kwargs)
            loss = self.loss_function(predictions, targets).mean()
            loss.backward()
            self.optimizer.step()

        return {Keys.IMAGE: inputs, Keys.LABEL: targets, Keys.PRED: predictions, Keys.LOSS: loss.item()}


class GanTrainer(Trainer):
    """
    Generative adversarial network training based on Goodfellow et al. 2014 https://arxiv.org/abs/1406.266,
    inherits from ``Trainer`` and ``Workflow``.

    Training Loop: for each batch of data size `m`
        1. Generate `m` fakes from random latent codes.
        2. Update discriminator with these fakes and current batch reals, repeated d_train_steps times.
        3. If g_update_latents, generate `m` fakes from new random latent codes.
        4. Update generator with these fakes using discriminator feedback.

    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for engine to run.
        train_data_loader: Core ignite engines uses `DataLoader` for training loop batchdata.
        g_network: generator (G) network architecture.
        g_optimizer: G optimizer function.
        g_loss_function: G loss function for optimizer.
        d_network: discriminator (D) network architecture.
        d_optimizer: D optimizer function.
        d_loss_function: D loss function for optimizer.
        epoch_length: number of iterations for one epoch, default to `len(train_data_loader)`.
        g_inferer: inference method to execute G model forward. Defaults to ``SimpleInferer()``.
        d_inferer: inference method to execute D model forward. Defaults to ``SimpleInferer()``.
        d_train_steps: number of times to update D with real data minibatch. Defaults to ``1``.
        latent_shape: size of G input latent code. Defaults to ``64``.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        d_prepare_batch: callback function to prepare batchdata for D inferer.
            Defaults to return ``GanKeys.REALS`` in batchdata dict.
        g_prepare_batch: callback function to create batch of latent input for G inferer.
            Defaults to return random latents.
        g_update_latents: Calculate G loss with new latent codes. Defaults to ``True``.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
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
        epoch_length: Optional[int] = None,
        g_inferer: Optional[Inferer] = None,
        d_inferer: Optional[Inferer] = None,
        d_train_steps: int = 1,
        latent_shape: int = 64,
        non_blocking: bool = False,
        d_prepare_batch: Callable = default_prepare_batch,
        g_prepare_batch: Callable = default_make_latent,
        g_update_latents: bool = True,
        iteration_update: Optional[Callable] = None,
        post_transform: Optional[Transform] = None,
        key_train_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        train_handlers: Optional[Sequence] = None,
    ):
        # set up Ignite engine and environments
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=d_prepare_batch,
            iteration_update=iteration_update,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            handlers=train_handlers,
            post_transform=post_transform,
        )
        self.g_network = g_network
        self.g_optimizer = g_optimizer
        self.g_loss_function = g_loss_function
        self.g_inferer = SimpleInferer() if g_inferer is None else g_inferer
        self.d_network = d_network
        self.d_optimizer = d_optimizer
        self.d_loss_function = d_loss_function
        self.d_inferer = SimpleInferer() if d_inferer is None else d_inferer
        self.d_train_steps = d_train_steps
        self.latent_shape = latent_shape
        self.g_prepare_batch = g_prepare_batch
        self.g_update_latents = g_update_latents

    def _iteration(
        self, engine: Engine, batchdata: Union[Dict, Sequence]
    ) -> Dict[str, Union[torch.Tensor, int, float, bool]]:
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

        d_input = self.prepare_batch(batchdata, engine.state.device, engine.non_blocking)
        batch_size = self.data_loader.batch_size
        g_input = self.g_prepare_batch(batch_size, self.latent_shape, engine.state.device, engine.non_blocking)
        g_output = self.g_inferer(g_input, self.g_network)

        # Train Discriminator
        d_total_loss = torch.zeros(
            1,
        )
        for _ in range(self.d_train_steps):
            self.d_optimizer.zero_grad()
            dloss = self.d_loss_function(g_output, d_input)
            dloss.backward()
            self.d_optimizer.step()
            d_total_loss += dloss.item()

        # Train Generator
        if self.g_update_latents:
            g_input = self.g_prepare_batch(batch_size, self.latent_shape, engine.state.device, engine.non_blocking)
        g_output = self.g_inferer(g_input, self.g_network)
        self.g_optimizer.zero_grad()
        g_loss = self.g_loss_function(g_output)
        g_loss.backward()
        self.g_optimizer.step()

        return {
            GanKeys.REALS: d_input,
            GanKeys.FAKES: g_output,
            GanKeys.LATENTS: g_input,
            GanKeys.GLOSS: g_loss.item(),
            GanKeys.DLOSS: d_total_loss.item(),
        }

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

from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from monai.config import IgniteInfo
from monai.engines.utils import (
    GanKeys,
    IterationEvents,
    default_make_latent,
    default_metric_cmp_fn,
    default_prepare_batch,
)
from monai.engines.workflow import Workflow
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import PT_BEFORE_1_7, min_version, optional_import
from monai.utils.enums import CommonKeys as Keys

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

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
        train_data_loader: Ignite engine use data_loader to run, must be Iterable or torch.DataLoader.
        network: network to train in the trainer, should be regular PyTorch `torch.nn.Module`.
        optimizer: the optimizer associated to the network, should be regular PyTorch optimizer from `torch.optim`
            or its subclass.
        loss_function: the loss function associated to the optimizer, should be regular PyTorch loss,
            which inherit from `torch.nn.modules.loss`.
        epoch_length: number of iterations for one epoch, default to `len(train_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_train_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_train_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        train_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        amp: whether to enable auto-mixed-precision training, default is False.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        optim_set_to_none: when calling `optimizer.zero_grad()`, instead of setting to zero, set the grads to None.
            more details: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html.

    """

    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        train_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        optimizer: Optimizer,
        loss_function: Callable,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        postprocessing: Optional[Transform] = None,
        key_train_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        train_handlers: Optional[Sequence] = None,
        amp: bool = False,
        event_names: Optional[List[Union[str, EventEnum]]] = None,
        event_to_attr: Optional[dict] = None,
        decollate: bool = True,
        optim_set_to_none: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            data_loader=train_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_metric=key_train_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            handlers=train_handlers,
            amp=amp,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
        )

        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.inferer = SimpleInferer() if inferer is None else inferer
        self.optim_set_to_none = optim_set_to_none

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
        # put iteration outputs into engine.state
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets}

        def _compute_pred_loss():
            engine.state.output[Keys.PRED] = self.inferer(inputs, self.network, *args, **kwargs)
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            engine.state.output[Keys.LOSS] = self.loss_function(engine.state.output[Keys.PRED], targets).mean()
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

        self.network.train()
        # `set_to_none` only work from PyTorch 1.7.0
        if PT_BEFORE_1_7:
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad(set_to_none=self.optim_set_to_none)

        if self.amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                _compute_pred_loss()
            self.scaler.scale(engine.state.output[Keys.LOSS]).backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            _compute_pred_loss()
            engine.state.output[Keys.LOSS].backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            self.optimizer.step()
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output


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
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_train_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_train_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        train_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        optim_set_to_none: when calling `optimizer.zero_grad()`, instead of setting to zero, set the grads to None.
            more details: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html.

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
        postprocessing: Optional[Transform] = None,
        key_train_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        train_handlers: Optional[Sequence] = None,
        decollate: bool = True,
        optim_set_to_none: bool = False,
    ):
        if not isinstance(train_data_loader, DataLoader):
            raise ValueError("train_data_loader must be PyTorch DataLoader.")

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
            metric_cmp_fn=metric_cmp_fn,
            handlers=train_handlers,
            postprocessing=postprocessing,
            decollate=decollate,
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
        self.optim_set_to_none = optim_set_to_none

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
        batch_size = self.data_loader.batch_size  # type: ignore
        g_input = self.g_prepare_batch(batch_size, self.latent_shape, engine.state.device, engine.non_blocking)
        g_output = self.g_inferer(g_input, self.g_network)

        # Train Discriminator
        d_total_loss = torch.zeros(
            1,
        )
        for _ in range(self.d_train_steps):
            # `set_to_none` only work from PyTorch 1.7.0
            if PT_BEFORE_1_7:
                self.d_optimizer.zero_grad()
            else:
                self.d_optimizer.zero_grad(set_to_none=self.optim_set_to_none)
            dloss = self.d_loss_function(g_output, d_input)
            dloss.backward()
            self.d_optimizer.step()
            d_total_loss += dloss.item()

        # Train Generator
        if self.g_update_latents:
            g_input = self.g_prepare_batch(batch_size, self.latent_shape, engine.state.device, engine.non_blocking)
        g_output = self.g_inferer(g_input, self.g_network)
        if PT_BEFORE_1_7:
            self.g_optimizer.zero_grad()
        else:
            self.g_optimizer.zero_grad(set_to_none=self.optim_set_to_none)
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

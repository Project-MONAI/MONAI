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

from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from monai.engines.utils import default_prepare_batch
from monai.transforms import apply_transform
from monai.utils import ensure_tuple, exact_version, optional_import

IgniteEngine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")
State, _ = optional_import("ignite.engine", "0.4.2", exact_version, "State")
Events, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "Metric")


class Workflow(IgniteEngine):  # type: ignore[valid-type, misc] # due to optional_import
    """
    Workflow defines the core work process inheriting from Ignite engine.
    All trainer, validator and evaluator share this same workflow as base class,
    because they all can be treated as same Ignite engine loops.
    It initializes all the sharable data in Ignite engine.state.
    And attach additional processing logics to Ignite engine based on Event-Handler mechanism.

    Users should consider to inherit from `trainer` or `evaluator` to develop more trainers or evaluators.

    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for engine to run, validator and evaluator have only 1 epoch.
        data_loader: Ignite engine use data_loader to run, must be torch.DataLoader.
        epoch_length: number of iterations for one epoch, default to `len(data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse image and label for every iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        post_transform: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        amp: whether to enable auto-mixed-precision training or inference, default is False.

    Raises:
        TypeError: When ``device`` is not a ``torch.Device``.
        TypeError: When ``data_loader`` is not a ``torch.utils.data.DataLoader``.
        TypeError: When ``key_metric`` is not a ``Optional[dict]``.
        TypeError: When ``additional_metrics`` is not a ``Optional[dict]``.

    """

    def __init__(
        self,
        device: torch.device,
        max_epochs: int,
        data_loader: DataLoader,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        post_transform: Optional[Callable] = None,
        key_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        handlers: Optional[Sequence] = None,
        amp: bool = False,
    ) -> None:
        if iteration_update is not None:
            super().__init__(iteration_update)
        else:
            super().__init__(self._iteration)
        if not isinstance(device, torch.device):
            raise TypeError(f"device must be a torch.device but is {type(device).__name__}.")
        if not isinstance(data_loader, DataLoader):
            raise TypeError(f"data_loader must be a torch.utils.data.DataLoader but is {type(data_loader).__name__}.")
        sampler = data_loader.__dict__["sampler"]
        if isinstance(sampler, DistributedSampler):

            @self.on(Events.EPOCH_STARTED)
            def set_sampler_epoch(engine: Engine):
                sampler.set_epoch(engine.state.epoch)

        # set all sharable data for the workflow based on Ignite engine.state
        self.state = State(
            rank=dist.get_rank() if dist.is_available() and dist.is_initialized() else 0,
            seed=0,
            iteration=0,
            epoch=0,
            max_epochs=max_epochs,
            epoch_length=len(data_loader) if epoch_length is None else epoch_length,
            output=None,
            batch=None,
            metrics={},
            dataloader=None,
            device=device,
            key_metric_name=None,  # we can set many metrics, only use key_metric to compare and save the best model
            best_metric=-1,
            best_metric_epoch=-1,
        )
        self.data_loader = data_loader
        self.non_blocking = non_blocking
        self.prepare_batch = prepare_batch

        if post_transform is not None:

            @self.on(Events.ITERATION_COMPLETED)
            def run_post_transform(engine: Engine) -> None:
                if post_transform is None:
                    raise AssertionError
                engine.state.output = apply_transform(post_transform, engine.state.output)

        if key_metric is not None:

            if not isinstance(key_metric, dict):
                raise TypeError(f"key_metric must be None or a dict but is {type(key_metric).__name__}.")
            self.state.key_metric_name = list(key_metric.keys())[0]
            metrics = key_metric
            if additional_metrics is not None and len(additional_metrics) > 0:
                if not isinstance(additional_metrics, dict):
                    raise TypeError(
                        f"additional_metrics must be None or a dict but is {type(additional_metrics).__name__}."
                    )
                metrics.update(additional_metrics)
            for name, metric in metrics.items():
                metric.attach(self, name)

            @self.on(Events.EPOCH_COMPLETED)
            def _compare_metrics(engine: Engine) -> None:
                if engine.state.key_metric_name is not None:
                    current_val_metric = engine.state.metrics[engine.state.key_metric_name]
                    if current_val_metric > engine.state.best_metric:
                        self.logger.info(f"Got new best metric of {engine.state.key_metric_name}: {current_val_metric}")
                        engine.state.best_metric = current_val_metric
                        engine.state.best_metric_epoch = engine.state.epoch

        if handlers is not None:
            handlers_ = ensure_tuple(handlers)
            for handler in handlers_:
                handler.attach(self)
        self.amp = amp

    def run(self) -> None:
        """
        Execute training, validation or evaluation based on Ignite Engine.

        """
        super().run(data=self.data_loader, max_epochs=self.state.max_epochs)

    def _iteration(self, engine: Engine, batchdata: Dict[str, torch.Tensor]):
        """
        Abstract callback function for the processing logic of 1 iteration in Ignite Engine.
        Need subclass to implement different logics, like SupervisedTrainer/Evaluator, GANTrainer, etc.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

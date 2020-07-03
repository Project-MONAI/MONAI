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

from typing import Callable, Optional
import torch

from monai.transforms import apply_transform
from monai.utils import exact_version, optional_import, ensure_tuple
from monai.engines.utils import default_prepare_batch

Engine, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Engine")
State, _ = optional_import("ignite.engine", "0.3.0", exact_version, "State")
Events, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Events")


class Workflow(Engine):
    """
    Workflow defines the core work process inheriting from Ignite engine.
    All trainer, validator and evaluator share this same workflow as base class,
    because they all can be treated as same Ignite engine loops.
    It initializes all the sharable data in Ignite engine.state.
    And attach additional processing logics to Ignite engine based on Event-Handler mechanism.

    Users should consider to inherit from `trainer` or `evaluator` to develop more trainers or evaluators.

    Args:
        device (torch.device): an object representing the device on which to run.
        max_epochs: the total epoch number for engine to run, validator and evaluator have only 1 epoch.
        amp: whether to enable auto-mixed-precision training, reserved.
        data_loader (torch.DataLoader): Ignite engine use data_loader to run, must be torch.DataLoader.
        prepare_batch: function to parse image and label for every iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        post_transform (Transform): execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_metric (ignite.metric): compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics (dict): more Ignite metrics that also attach to Ignite Engine.
        handlers (list): every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.

    """

    def __init__(
        self,
        device,
        max_epochs: int,
        amp: bool,
        data_loader,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        post_transform=None,
        key_metric=None,
        additional_metrics=None,
        handlers=None,
    ):
        # pytype: disable=invalid-directive
        # pytype: disable=wrong-arg-count
        super().__init__(iteration_update if iteration_update is not None else self._iteration)
        # pytype: enable=invalid-directive
        # pytype: enable=wrong-arg-count
        # FIXME:
        if amp:
            self.logger.info("Will add AMP support when PyTorch v1.6 released.")
        if not isinstance(device, torch.device):
            raise ValueError("device must be PyTorch device object.")
        if not isinstance(data_loader, torch.utils.data.DataLoader):  # type: ignore
            raise ValueError("data_loader must be PyTorch DataLoader.")

        # set all sharable data for the workflow based on Ignite engine.state
        self.state = State(
            seed=0,
            iteration=0,
            epoch=0,
            max_epochs=max_epochs,
            epoch_length=-1,
            output=None,
            batch=None,
            metrics={},
            dataloader=None,
            device=device,
            amp=amp,
            key_metric_name=None,  # we can set many metrics, only use key_metric to compare and save the best model
            best_metric=-1,
            best_metric_epoch=-1,
        )
        self.data_loader = data_loader
        self.prepare_batch = prepare_batch

        if post_transform is not None:

            @self.on(Events.ITERATION_COMPLETED)
            def run_post_transform(engine):
                engine.state.output = apply_transform(post_transform, engine.state.output)

        if key_metric is not None:

            if not isinstance(key_metric, dict):
                raise ValueError("key_metric must be a dict object.")
            self.state.key_metric_name = list(key_metric.keys())[0]
            metrics = key_metric
            if additional_metrics is not None and len(additional_metrics) > 0:
                if not isinstance(additional_metrics, dict):
                    raise ValueError("additional_metrics must be a dict object.")
                metrics.update(additional_metrics)
            for name, metric in metrics.items():
                metric.attach(self, name)

            @self.on(Events.EPOCH_COMPLETED)
            def _compare_metrics(engine):
                if engine.state.key_metric_name is not None:
                    current_val_metric = engine.state.metrics[engine.state.key_metric_name]
                    if current_val_metric > engine.state.best_metric:
                        self.logger.info(f"Got new best metric of {engine.state.key_metric_name}: {current_val_metric}")
                        engine.state.best_metric = current_val_metric
                        engine.state.best_metric_epoch = engine.state.epoch

        if handlers is not None:
            handlers = ensure_tuple(handlers)
            for handler in handlers:
                handler.attach(self)

    def run(self) -> None:
        """
        Execute training, validation or evaluation based on Ignite Engine.

        """
        super().run(data=self.data_loader, epoch_length=len(self.data_loader))

    def _iteration(self, engine: Engine, batchdata):
        """
        Abstract callback function for the processing logic of 1 iteration in Ignite Engine.
        Need subclass to implement different logics, like SupervisedTrainer/Evaluator, GANTrainer, etc.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata (TransformContext, ndarray): input data for this iteration.

        Raises:
            NotImplementedError: Subclass {self.__class__.__name__} must implement the compute method

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the compute method")

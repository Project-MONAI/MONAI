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
from monai.inferers import SimpleInferer
from monai.utils import exact_version, optional_import

from monai.engines.utils import CommonKeys as Keys
from monai.engines.utils import default_prepare_batch
from monai.engines.workflow import Workflow

Engine, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Engine")
Metric, _ = optional_import("ignite.metrics", "0.3.0", exact_version, "Metric")


class Evaluator(Workflow):
    """
    Base class for all kinds of evaluators, inherits from Workflow.

    Args:
        device (torch.device): an object representing the device on which to run.
        val_data_loader (torch.DataLoader): Ignite engine use data_loader to run, must be torch.DataLoader.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        post_transform (Transform): execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric (ignite.metric): compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics (dict): more Ignite metrics that also attach to Ignite Engine.
        val_handlers (list): every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        post_transform=None,
        key_val_metric: Optional[Metric] = None,
        additional_metrics=None,
        val_handlers=None,
    ):
        super().__init__(
            device=device,
            max_epochs=1,
            amp=False,
            data_loader=val_data_loader,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            post_transform=post_transform,
            key_metric=key_val_metric,
            additional_metrics=additional_metrics,
            handlers=val_handlers,
        )

    def run(self, global_epoch: int = 1):
        """
        Execute validation/evaluation based on Ignite Engine.

        Args:
            global_epoch: the overall epoch if during a training. evaluator engine can get it from trainer.

        """
        # init env value for current validation process
        self.state.max_epochs = global_epoch
        self.state.epoch = global_epoch - 1
        self.state.iteration = 0
        super().run()

    def get_validation_stats(self):
        return {"best_validation_metric": self.state.best_metric, "best_validation_epoch": self.state.best_metric_epoch}


class SupervisedEvaluator(Evaluator):
    """
    Standard supervised evaluation method with image and label(optional), inherits from evaluator and Workflow.

    Args:
        device (torch.device): an object representing the device on which to run.
        val_data_loader (torch.DataLoader): Ignite engine use data_loader to run, must be torch.DataLoader.
        network (Network): use the network to run model forward.
        prepare_batch: function to parse image and label for current iteration.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        inferer (Inferer): inference method that execute model forward on input data, like: SlidingWindow, etc.
        post_transform (Transform): execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric (ignite.metric): compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics (dict): more Ignite metrics that also attach to Ignite Engine.
        val_handlers (list): every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader,
        network,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer=SimpleInferer(),
        post_transform=None,
        key_val_metric=None,
        additional_metrics=None,
        val_handlers=None,
    ):
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            post_transform=post_transform,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            val_handlers=val_handlers,
        )

        self.network = network
        self.inferer = inferer

    def _iteration(self, engine: Engine, batchdata):
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata (TransformContext, ndarray): input data for this iteration.

        Raises:
            ValueError: must provide batch data for current iteration.

        """
        if batchdata is None:
            raise ValueError("must provide batch data for current iteration.")
        inputs, targets = self.prepare_batch(batchdata)
        inputs = inputs.to(engine.state.device)
        if targets is not None:
            targets = targets.to(engine.state.device)

        # execute forward computation
        self.network.eval()
        with torch.no_grad():
            predictions = self.inferer(inputs, self.network)

        return {Keys.IMAGE: inputs, Keys.LABEL: targets, Keys.PRED: predictions}

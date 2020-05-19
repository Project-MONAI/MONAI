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

from abc import abstractmethod
import torch
from monai.inferers.inferer import RegularInferer
from .workflow import Workflow
from .utils import default_prepare_batch
from .utils import CommonKeys as Keys


class Evaluator(Workflow):
    """Base class for all kinds of evaluators, extends from Workflow.

    Args:
        device (torch.device): an object representing the device on which to run.
        val_data_loader (torch.DataLoader): Ignite engine use data_loader to run, must be torch.DataLoader.
        prepare_batch (Callable): function to parse image and label for current iteration.
        key_metric (ignite.metric): compute metric when every iteration completed, and save average
            value to engine.state.metrics when epoch completed. also use key_metric to select and
            save checkpoint into files.
        additional_metrics (list): more ignite metrics that also attach to Ignite Engine.
        handlers (list): every handler is a set of Ignite Event-Handlers, like:
            CheckpointHandler, StatsHandler, TimerHandler, etc.

    """

    def __init__(
        self,
        device,
        val_data_loader,
        prepare_batch=default_prepare_batch,
        key_metric=None,
        additional_metrics=None,
        handlers=None,
    ):
        super().__init__(device, 1, False, val_data_loader, prepare_batch, key_metric, additional_metrics, handlers)

    def evaluate(self, global_epoch=1):
        """Execute validation/evaluation based on Ignite Engine.

        """
        # init env value for current validation process
        self.engine.state.max_epochs = global_epoch
        self.engine.state.epoch = global_epoch - 1
        self._run()

    def get_validation_stats(self):
        return {
            "best_validation_metric": self.engine.state.best_metric,
            "best_validation_epoch": self.engine.state.best_metric_epoch,
        }

    @abstractmethod
    def _iteration(self, engine, batchdata):
        raise NotImplementedError("Subclass {} must implement the compute method".format(self.__class__.__name__))


class SupervisedEvaluator(Evaluator):
    """Standard supervised evaluation method with image and label, extends from evaluator and Workflow.

    Args:
        device (torch.device): an object representing the device on which to run.
        val_data_loader (torch.DataLoader): Ignite engine use data_loader to run, must be torch.DataLoader.
        network (Network): use the network to run model forward.
        prepare_batch (Callable): function to parse image and label for current iteration.
        inferer (Inferer): inference method that execute model forward on input data, like: SlidingWindow, etc.
        val_handlers (list): every handler is a set of Ignite Event-Handlers, like:
                             CheckpointHandler, StatslogHandler, TimerHandler, etc.
        key_val_metric (ignite.metric): compute metric when every iteration completed,
                                    and save average value to engine.state.metrics when epoch completed.
                                    also use key_metric to select and save checkpoint into files.
        additional_metrics (list): more ignite metrics that also attach to Ignite Engine.

    Note:
        the "batch_size" here is to run a batch of window slices of 1 input image, not batch size of input images.

    """

    def __init__(
        self,
        device,
        val_data_loader,
        network,
        prepare_batch=default_prepare_batch,
        inferer=RegularInferer(),
        val_handlers=None,
        key_val_metric=None,
        additional_metrics=None,
    ):
        super().__init__(device, val_data_loader, prepare_batch, key_val_metric, additional_metrics, val_handlers)

        self.network = network
        self.inferer = inferer

    def _iteration(self, engine, batchdata):
        """callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata (TransformContext, ndarray): input data for this iteration.

        """
        assert batchdata is not None, "must provide batch data for current iteraion."
        inputs, targets = self.prepare_batch(batchdata)
        inputs = inputs.to(engine.state.device)
        if targets is not None:
            targets = targets.to(engine.state.device)

        # execute forward computation
        self.network.eval()
        with torch.no_grad():
            predictions = self.inferer(inputs, self.network)

        return {Keys.Y_PRED: predictions, Keys.Y: targets, Keys.INFO: None}

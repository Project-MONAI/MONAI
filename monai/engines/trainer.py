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
from monai.inferers.inferer import RegularInferer
from .workflow import Workflow
from .utils import default_prepare_batch
from .utils import CommonKeys as Keys


class Trainer(Workflow):
    """
    Base class for all kinds of trainers, extends from Workflow.

    """

    def train(self):
        """
        Execute training based on Ignite Engine.

        """
        self._run()

    def get_train_stats(self):
        return {"total_epochs": self.state.max_epochs, "total_iterations": self.state.epoch_length}

    @abstractmethod
    def _iteration(self, engine, batchdata):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the compute method")


class SupervisedTrainer(Trainer):
    """Standard supervised training method with image and label, extends from trainer and Workflow.

    Args:
        device (torch.device): an object representing the device on which to run.
        max_epochs (Int): the total epoch number for engine to run, validator and evaluator have only 1 epoch.
        train_data_loader (torch.DataLoader): Ignite engine use data_loader to run, must be torch.DataLoader.
        network (Network): to train with this network.
        optimizer (Optimizer): the optimizer associated to the network.
        loss_function (Loss): the loss function associated to the optimizer.
        prepare_batch (Callable): function to parse image and label for current iteration.
        iteration_update (Callable): the callable function for every iteration, expect to accept `engine`
            and `batchdata` as input parameters. if not provided, use `self._iteration()` instead.
        lr_scheduler (LR Scheduler): the lr scheduler associated to the optimizer.
        inferer (Inferer): inference method that execute model forward on input data, like: SlidingWindow, etc.
        train_handlers (list): every handler is a set of Ignite Event-Handlers, like:
            CheckpointHandler, StatslogHandler, TimerHandler, etc.
        amp (Bool): whether to enable auto-mixed-precision training.
        key_train_metric (ignite.metric): compute metric when every iteration completed, and save average
            value to engine.state.metrics when epoch completed. also use key_metric to select and save
            checkpoint into files.
        additional_metrics (list): more ignite metrics that also attach to Ignite Engine.
        val_interval (Int): do validation every N epochs during training, disable validation if N = 0.
        validator (Evaluator): run the validator when trigger validation, suppose to be Evaluator.

    """

    def __init__(
        self,
        device,
        max_epochs,
        train_data_loader,
        network,
        optimizer,
        loss_function,
        prepare_batch=default_prepare_batch,
        iteration_update=None,
        lr_scheduler=None,
        inferer=RegularInferer(),
        train_handlers=None,
        amp=True,
        key_train_metric=None,
        additional_metrics=None
    ):
        # set up Ignite engine and environments
        super().__init__(
            device,
            max_epochs,
            amp,
            train_data_loader,
            prepare_batch,
            key_train_metric,
            additional_metrics,
            train_handlers,
            iteration_update
        )

        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.inferer = inferer

    def _iteration(self, engine, batchdata):
        """callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata (dict or array of tensor): input data for this iteration.

        """
        assert batchdata is not None, "must provide batch data for current iteraion."
        inputs, targets = self.prepare_batch(batchdata)
        inputs, targets = inputs.to(engine.state.device), targets.to(engine.state.device)

        results = dict()
        self.network.train()
        self.optimizer.zero_grad()
        # execute forward computation
        predictions = self.inferer(inputs, self.network)
        # compute loss
        loss = self.loss_function(predictions, targets).mean()
        loss.backward()
        results[Keys.LOSS] = loss.item()
        self.optimizer.step()

        return {Keys.PRED: predictions, Keys.LABEL: targets, Keys.INFO: results}

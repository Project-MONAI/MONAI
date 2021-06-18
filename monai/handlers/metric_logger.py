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

from collections import defaultdict
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING, Callable, DefaultDict, List, Optional

from monai.utils import exact_version, optional_import
from monai.utils.enums import CommonKeys

Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")


def _get_loss_from_output(output, loss_key: str = CommonKeys.LOSS):
    return output[loss_key].item()


class MetricLoggerKeys(Enum):
    METRICS = "Metrics"
    LOSS = "Loss"


class MetricLogger:
    """
    Collect per-iteration metrics and loss value from the attached trainer. This will also collect metric values from
    a given evaluator object which is expected to perform evaluation at the end of training epochs. This class is
    useful for collecting loss and metric values in one place for storage with checkpoint savers (`state_dict` and
    `load_state_dict` methods provided as expected by Pytorch and Ignite) and for graphing during training.

    Example::
        # construct an evaluator saving mean dice metric values in the key "val_mean_dice"
        evaluator = SupervisedEvaluator(..., key_val_metric={"val_mean_dice": MeanDice(...)})

        # construct the logger and associate with evaluator to extract metric values from
        logger = MetricLogger(evaluator=evaluator)

        # construct the trainer with the logger passed in as a handler so that it logs loss values
        trainer = SupervisedTrainer(..., train_handlers=[logger, ValidationHandler(1, evaluator)])

        # run training, logger.loss will be a list of (iteration, loss) values, logger.metrics a dict with key
        # "val_mean_dice" storing a list of (iteration, metric) values
        trainer.run()

    Args:
        loss_transform: Converts the `output` value from the trainer's state into a loss value
        metric_transform: Converts the metric value coming from the trainer/evaluator's state into a storable value
        evaluator: Optional evaluator to consume metric results from at the end of its evaluation run
    """

    def __init__(
        self,
        loss_transform: Callable = _get_loss_from_output,
        metric_transform: Callable = lambda x: x,
        evaluator: Optional[Engine] = None,
    ) -> None:
        self.loss_transform = loss_transform
        self.metric_transform = metric_transform
        self.loss: List = []
        self.metrics: DefaultDict = defaultdict(list)
        self.iteration = 0
        self.lock = RLock()

        if evaluator is not None:
            self.attach_evaluator(evaluator)

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def attach_evaluator(self, evaluator: Engine) -> None:
        """
        Attach event  handlers to the given evaluator to log metric values from it.

        Args:
            evaluator: Ignite Engine implementing network evaluation
        """
        evaluator.add_event_handler(Events.COMPLETED, self.log_metrics)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        with self.lock:
            self.iteration = engine.state.iteration
            lossval = self.loss_transform(engine.state.output)

            self.loss.append((self.iteration, lossval))
            self.log_metrics(engine)

    def log_metrics(self, engine: Engine) -> None:
        """
        Log metrics from the given Engine's state member.

        Args:
            engine: Ignite Engine to log from
        """
        with self.lock:
            for m, v in engine.state.metrics.items():
                v = self.metric_transform(v)
                self.metrics[m].append((self.iteration, v))

    def state_dict(self):
        return {MetricLoggerKeys.LOSS: self.loss, MetricLoggerKeys.METRICS: self.metrics}

    def load_state_dict(self, state_dict):
        self.loss[:] = state_dict[MetricLoggerKeys.LOSS]
        self.metrics.clear()
        self.metrics.update(state_dict[MetricLoggerKeys.METRICS])


metriclogger = MetricLogger

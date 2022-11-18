# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import torch

from monai.config import IgniteInfo
from monai.utils import min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
mlflow, _ = optional_import("mlflow")

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine", as_type="decorator"
    )

DEFAULT_TAG = "Loss"


class MLFlowHandler:
    """
    MLFlowHandler defines a set of Ignite Event-handlers for the MLFlow tracking logics.
    It can be used for any Ignite Engine(trainer, validator and evaluator).
    And it can track both epoch level and iteration level logging, then MLFlow can store
    the data and visualize.
    The expected data source is Ignite ``engine.state.output`` and ``engine.state.metrics``.

    Default behaviors:
        - When EPOCH_COMPLETED, track each dictionary item in
          ``engine.state.metrics`` in MLFlow.
        - When ITERATION_COMPLETED, track expected item in
          ``self.output_transform(engine.state.output)`` in MLFlow, default to `Loss`.

    Usage example is available in the tutorial:
    https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unet_segmentation_3d_ignite.ipynb.

    Args:
        tracking_uri: connects to a tracking URI. can also set the `MLFLOW_TRACKING_URI` environment
            variable to have MLflow find a URI from there. in both cases, the URI can either be
            a HTTP/HTTPS URI for a remote server, a database connection string, or a local path
            to log data to a directory. The URI defaults to path `mlruns`.
            for more details: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri.
        iteration_log: whether to log data to MLFlow when iteration completed, default to `True`.
        epoch_log: whether to log data to MLFlow when epoch completed, default to `True`.
        epoch_logger: customized callable logger for epoch level logging with MLFlow.
            Must accept parameter "engine", use default logger if None.
        iteration_logger: customized callable logger for iteration level logging with MLFlow.
            Must accept parameter "engine", use default logger if None.
        output_transform: a callable that is used to transform the
            ``ignite.engine.state.output`` into a scalar to track, or a dictionary of {key: scalar}.
            By default this value logging happens when every iteration completed.
            The default behavior is to track loss from output[0] as output is a decollated list
            and we replicated loss value for every item of the decollated list.
            `engine.state` and `output_transform` inherit from the ignite concept:
            https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
            https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
        global_epoch_transform: a callable that is used to customize global epoch number.
            For example, in evaluation, the evaluator engine might want to track synced epoch number
            with the trainer engine.
        state_attributes: expected attributes from `engine.state`, if provided, will extract them
            when epoch completed.
        tag_name: when iteration output is a scalar, `tag_name` is used to track, defaults to `'Loss'`.

    For more details of MLFlow usage, please refer to: https://mlflow.org/docs/latest/index.html.

    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        iteration_log: bool = True,
        epoch_log: bool = True,
        epoch_logger: Optional[Callable[[Engine], Any]] = None,
        iteration_logger: Optional[Callable[[Engine], Any]] = None,
        output_transform: Callable = lambda x: x[0],
        global_epoch_transform: Callable = lambda x: x,
        state_attributes: Optional[Sequence[str]] = None,
        tag_name: str = DEFAULT_TAG,
    ) -> None:
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        self.iteration_log = iteration_log
        self.epoch_log = epoch_log
        self.epoch_logger = epoch_logger
        self.iteration_logger = iteration_logger
        self.output_transform = output_transform
        self.global_epoch_transform = global_epoch_transform
        self.state_attributes = state_attributes
        self.tag_name = tag_name

    def attach(self, engine: Engine) -> None:
        """
        Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if not engine.has_event_handler(self.start, Events.STARTED):
            engine.add_event_handler(Events.STARTED, self.start)
        if self.iteration_log and not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        if self.epoch_log and not engine.has_event_handler(self.epoch_completed, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_completed)

    def start(self) -> None:
        """
        Check MLFlow status and start if not active.

        """
        if mlflow.active_run() is None:
            mlflow.start_run()

    def close(self) -> None:
        """
        Stop current running logger of MLFlow.

        """
        mlflow.end_run()

    def epoch_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation epoch completed Event.
        Track epoch level log, default values are from Ignite `engine.state.metrics` dict.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.epoch_logger is not None:
            self.epoch_logger(engine)
        else:
            self._default_epoch_log(engine)

    def iteration_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation iteration completed Event.
        Track iteration level log.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.iteration_logger is not None:
            self.iteration_logger(engine)
        else:
            self._default_iteration_log(engine)

    def _default_epoch_log(self, engine: Engine) -> None:
        """
        Execute epoch level log operation.
        Default to track the values from Ignite `engine.state.metrics` dict and
        track the values of specified attributes of `engine.state`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        log_dict = engine.state.metrics
        if not log_dict:
            return

        current_epoch = self.global_epoch_transform(engine.state.epoch)
        mlflow.log_metrics(log_dict, step=current_epoch)

        if self.state_attributes is not None:
            attrs = {attr: getattr(engine.state, attr, None) for attr in self.state_attributes}
            mlflow.log_metrics(attrs, step=current_epoch)

    def _default_iteration_log(self, engine: Engine) -> None:
        """
        Execute iteration log operation based on Ignite `engine.state.output` data.
        Log the values from `self.output_transform(engine.state.output)`.
        Since `engine.state.output` is a decollated list and we replicated the loss value for every item
        of the decollated list, the default behavior is to track the loss from `output[0]`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        loss = self.output_transform(engine.state.output)
        if loss is None:
            return

        if not isinstance(loss, dict):
            loss = {self.tag_name: loss.item() if isinstance(loss, torch.Tensor) else loss}

        mlflow.log_metrics(loss, step=engine.state.iteration)

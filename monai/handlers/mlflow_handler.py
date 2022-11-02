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

import time
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

import torch

from monai.config import IgniteInfo
from monai.engines import Trainer
from monai.handlers.validation_handler import ValidationHandler
from monai.utils import min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
mlflow, _ = optional_import("mlflow")

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")

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
        experiment_name: name for an experiment, defaults to `default_experiment`.
        run_name: name for run in an experiment, defaults to `test_run`.
        experiment_param: a dict recording parameters which will not change through whole experiment,
            like torch version, cuda version and so on.
        artifacts: paths to images that need to be recorded after a whole run.
        optimizer_param_names: parameters' name in optimizer that need to be record during runing,
            defaults to `["lr"]`.

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
        experiment_name: str = "default_experiment",
        run_name: Optional[str] = None,
        experiment_param: Optional[Dict] = None,
        artifacts: Optional[Sequence[Path]] = None,
        optimizer_param_names: Sequence[str] = ["lr"],
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
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.experiment_param = experiment_param
        self.artifacts = artifacts
        self.optimizer_param_names = optimizer_param_names
        self.default_attr_name = ["seed", "max_epochs", "epoch_length"]
        self.client = mlflow.MlflowClient()

    def _try_log_param(self, engine: Engine, attr: str):
        engine_attr = getattr(engine, attr, None)
        if engine_attr:
            attr_type_string = str(type(engine_attr))
            mlflow.log_param(key=attr, value=attr_type_string)

    def _is_param_exists(self, param_name: str):
        cur_run = mlflow.active_run()
        log_data = self.client.get_run(cur_run.info.run_id).data
        param_dict = log_data.params
        if param_name in param_dict:
            return True
        else:
            return False

    def _delete_exist_param_in_dict(self, param_dict):
        key_list = [x for x in param_dict.keys()]
        for key in key_list:
            if self._is_param_exists(key):
                del param_dict[key]

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
        if not engine.has_event_handler(self.complete, Events.COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.complete)

    def start(self, engine: Engine) -> None:
        """
        Check MLFlow status and start if not active.

        """
        mlflow.set_experiment(self.experiment_name)
        if mlflow.active_run() is None:
            if self.run_name:
                mlflow.start_run(run_name=self.run_name)
            else:
                cur_time = time.strftime("%Y%m%d_%H%M%S")
                run_name = "run_" + cur_time
                mlflow.start_run(run_name=run_name)

        if self.experiment_param:
            mlflow.log_params(self.experiment_param)

        attrs = {attr: getattr(engine.state, attr, None) for attr in self.default_attr_name}
        self._delete_exist_param_in_dict(attrs)
        mlflow.log_params(attrs)

        default_log_param_list = ["network", "device", "optimizer", "loss_function"]
        for param_name in default_log_param_list:
            if self._is_param_exists(param_name):
                continue
            self._try_log_param(engine, param_name)

    def _parse_artifacts(self):
        artifact_list = []
        for path_name in self.artifacts:
            if os.path.isfile(path_name):
                artifact_list.append(path_name)
            else:
                for root, _, filenames in os.walk(path_name):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        artifact_list.append(file_path)
        return artifact_list

    def complete(self) -> None:
        """
        Handler for train or validation/evaluation completed Event.
        """
        if self.artifacts:
            artifact_list = self._parse_artifacts()
            for artifact in artifact_list:
                mlflow.log_artifact(artifact)

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

        events = engine._event_handlers

        for e in events:
            for handler, _, _ in engine._event_handlers[e]:
                if isinstance(handler, ValidationHandler):
                    evaluator_state = getattr(handler.validator, "state", None)
                    if evaluator_state:
                        handler_metrics_dict = evaluator_state.metrics
                        mlflow.log_metrics(handler_metrics_dict, step=current_epoch)

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

        if isinstance(engine, Trainer):
            mlflow.log_metrics(loss, step=engine.state.iteration)

        # If there is optimizer attr in engine, then record parameters specified in init function.
        try:
            cur_optimizer = engine.optimizer  # type: ignore
            for param_name in self.optimizer_param_names:
                params = {
                    f"{param_name} group_{i}": float(param_group[param_name])
                    for i, param_group in enumerate(cur_optimizer.param_groups)
                }
                mlflow.log_metrics(params, step=engine.state.iteration)

        except AttributeError:
            pass

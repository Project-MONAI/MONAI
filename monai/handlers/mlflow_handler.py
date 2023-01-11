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

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from monai.config import IgniteInfo
from monai.utils import ensure_tuple, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
mlflow, _ = optional_import("mlflow")
mlflow.entities, _ = optional_import("mlflow.entities")

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
            an HTTP/HTTPS URI for a remote server, a database connection string, or a local path
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
        run_name: name for run in an experiment.
        experiment_param: a dict recording parameters which will not change through whole experiment,
            like torch version, cuda version and so on.
        artifacts: paths to images that need to be recorded after a whole run.
        optimizer_param_names: parameters' name in optimizer that need to be record during running,
            defaults to "lr".
        close_on_complete: whether to close the mlflow run in `complete` phase in workflow, default to False.

    For more details of MLFlow usage, please refer to: https://mlflow.org/docs/latest/index.html.

    """

    # parameters that are logged at the start of training
    default_tracking_params = ["max_epochs", "epoch_length"]

    def __init__(
        self,
        tracking_uri: str | None = None,
        iteration_log: bool = True,
        epoch_log: bool = True,
        epoch_logger: Callable[[Engine], Any] | None = None,
        iteration_logger: Callable[[Engine], Any] | None = None,
        output_transform: Callable = lambda x: x[0],
        global_epoch_transform: Callable = lambda x: x,
        state_attributes: Sequence[str] | None = None,
        tag_name: str = DEFAULT_TAG,
        experiment_name: str = "monai_experiment",
        run_name: str | None = None,
        experiment_param: dict | None = None,
        artifacts: str | Sequence[Path] | None = None,
        optimizer_param_names: str | Sequence[str] = "lr",
        close_on_complete: bool = False,
    ) -> None:
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
        self.artifacts = ensure_tuple(artifacts)
        self.optimizer_param_names = ensure_tuple(optimizer_param_names)
        self.client = mlflow.MlflowClient(tracking_uri=tracking_uri if tracking_uri else None)
        self.close_on_complete = close_on_complete
        self.experiment = None
        self.cur_run = None

    def _delete_exist_param_in_dict(self, param_dict: dict) -> None:
        """
        Delete parameters in given dict, if they are already logged by current mlflow run.

        Args:
            param_dict: parameter dict to be logged to mlflow.
        """
        if self.cur_run is None:
            return

        key_list = list(param_dict.keys())
        log_data = self.client.get_run(self.cur_run.info.run_id).data
        log_param_dict = log_data.params
        for key in key_list:
            if key in log_param_dict:
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
            engine.add_event_handler(Events.COMPLETED, self.complete)
        if self.close_on_complete and (not engine.has_event_handler(self.close, Events.COMPLETED)):
            engine.add_event_handler(Events.COMPLETED, self.close)

    def start(self, engine: Engine) -> None:
        """
        Check MLFlow status and start if not active.

        """
        self._set_experiment()
        if not self.experiment:
            raise ValueError(f"Failed to set experiment '{self.experiment_name}' as the active experiment")

        if not self.cur_run:
            run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}" if self.run_name is None else self.run_name
            runs = self.client.search_runs(self.experiment.experiment_id)
            runs = [r for r in runs if r.info.run_name == run_name or not self.run_name]
            if runs:
                self.cur_run = self.client.get_run(runs[-1].info.run_id)  # pick latest active run
            else:
                self.cur_run = self.client.create_run(experiment_id=self.experiment.experiment_id, run_name=run_name)

        if self.experiment_param:
            self._log_params(self.experiment_param)

        attrs = {attr: getattr(engine.state, attr, None) for attr in self.default_tracking_params}
        self._delete_exist_param_in_dict(attrs)
        self._log_params(attrs)

    def _set_experiment(self):
        experiment = self.experiment
        if not experiment:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if not experiment:
                experiment_id = self.client.create_experiment(self.experiment_name)
                experiment = self.client.get_experiment(experiment_id)

        if experiment.lifecycle_stage != mlflow.entities.LifecycleStage.ACTIVE:
            raise ValueError(f"Cannot set a deleted experiment '{self.experiment_name}' as the active experiment")
        self.experiment = experiment

    def _log_params(self, params: dict[str, Any]) -> None:
        if not self.cur_run:
            raise ValueError("Current Run is not Active to log params")
        params_arr = [mlflow.entities.Param(key, str(value)) for key, value in params.items()]
        self.client.log_batch(run_id=self.cur_run.info.run_id, metrics=[], params=params_arr, tags=[])

    def _log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.cur_run:
            raise ValueError("Current Run is not Active to log metrics")

        run_id = self.cur_run.info.run_id
        timestamp = int(time.time() * 1000)
        metrics_arr = [mlflow.entities.Metric(key, value, timestamp, step or 0) for key, value in metrics.items()]
        self.client.log_batch(run_id=run_id, metrics=metrics_arr, params=[], tags=[])

    def _parse_artifacts(self):
        """
        Log artifacts to mlflow. Given a path, all files in the path will be logged recursively.
        Given a file, it will be logged to mlflow.
        """
        artifact_list = []
        for path_name in self.artifacts:
            # in case the input is (None,) by default
            if not path_name:
                continue
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
        if self.artifacts and self.cur_run:
            artifact_list = self._parse_artifacts()
            for artifact in artifact_list:
                self.client.log_artifact(self.cur_run.info.run_id, artifact)

    def close(self) -> None:
        """
        Stop current running logger of MLFlow.

        """
        if self.cur_run:
            status = mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.FINISHED)
            self.client.set_terminated(self.cur_run.info.run_id, status)
            self.cur_run = None

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
        self._log_metrics(log_dict, step=current_epoch)

        if self.state_attributes is not None:
            attrs = {attr: getattr(engine.state, attr, None) for attr in self.state_attributes}
            self._log_metrics(attrs, step=current_epoch)

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

        self._log_metrics(loss, step=engine.state.iteration)

        # If there is optimizer attr in engine, then record parameters specified in init function.
        if hasattr(engine, "optimizer"):
            cur_optimizer = engine.optimizer
            for param_name in self.optimizer_param_names:
                params = {
                    f"{param_name} group_{i}": float(param_group[param_name])
                    for i, param_group in enumerate(cur_optimizer.param_groups)
                }
                self._log_metrics(params, step=engine.state.iteration)

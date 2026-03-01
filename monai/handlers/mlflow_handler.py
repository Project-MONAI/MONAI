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
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from monai.apps.utils import get_logger
from monai.utils import CommonKeys, IgniteInfo, ensure_tuple, flatten_dict, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
mlflow, _ = optional_import("mlflow", descriptor="Please install mlflow before using MLFlowHandler.")
mlflow.entities, _ = optional_import(
    "mlflow.entities", descriptor="Please install mlflow.entities before using MLFlowHandler."
)
MlflowException, _ = optional_import(
    "mlflow.exceptions", name="MlflowException", descriptor="Please install mlflow before using MLFlowHandler."
)
pandas, _ = optional_import("pandas", descriptor="Please install pandas for recording the dataset.")
tqdm, _ = optional_import("tqdm", "4.47.0", min_version, "tqdm")

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine", as_type="decorator"
    )

DEFAULT_TAG = "Loss"

logger = get_logger(module_name=__name__)


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
            ``iteration_log`` can be also a function and it will be interpreted as an event filter
            (see https://pytorch.org/ignite/generated/ignite.engine.events.Events.html for details).
            Event filter function accepts as input engine and event value (iteration) and should return True/False.
        epoch_log: whether to log data to MLFlow when epoch completed, default to `True`.
            ``epoch_log`` can be also a function and it will be interpreted as an event filter.
            See ``iteration_log`` argument for more details.
        epoch_logger: customized callable logger for epoch level logging with MLFlow.
            Must accept parameter "engine", use default logger if None.
        iteration_logger: customized callable logger for iteration level logging with MLFlow.
            Must accept parameter "engine", use default logger if None.
        dataset_logger: customized callable logger to log the dataset information with MLFlow.
            Must accept parameter "dataset_dict", use default logger if None.
        dataset_dict: a dictionary in which the key is the name of the dataset and the value is a PyTorch
            dataset, that needs to be recorded. This arg is only useful when MLFlow version >= 2.4.0.
            For more details about how to log data with MLFlow, please go to the website:
            https://mlflow.org/docs/latest/python_api/mlflow.data.html.
        dataset_keys: a key or a collection of keys to indicate contents in the dataset that
            need to be stored by MLFlow.
        output_transform: a callable that is used to transform the
            ``ignite.engine.state.output`` into a scalar to track, or a dictionary of {key: scalar}.
            By default this value logging happens when every iteration completed.
            The default behavior is to track loss from output[0] as output is a decollated list
            and we replicated loss value for every item of the decollated list.
            `engine.state` and `output_transform` inherit from the ignite concept:
            https://pytorch-ignite.ai/concepts/03-state/, explanation and usage example are in the tutorial:
            https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
        global_epoch_transform: a callable that is used to customize global epoch number.
            For example, in evaluation, the evaluator engine might want to track synced epoch number
            with the trainer engine.
        state_attributes: expected attributes from `engine.state`, if provided, will extract them
            when epoch completed.
        tag_name: when iteration output is a scalar, `tag_name` is used to track, defaults to `'Loss'`.
        experiment_name: the experiment name of MLflow, default to `'monai_experiment'`. An experiment can be
            used to record several runs.
        run_name: the run name in an experiment. A run can be used to record information about a workflow,
            like the loss, metrics and so on.
        experiment_param: a dict recording parameters which will not change through the whole workflow,
            like torch version, cuda version and so on.
        artifacts: paths to images that need to be recorded after running the workflow.
        optimizer_param_names: parameter names in the optimizer that need to be recorded during running the
            workflow, default to `'lr'`.
        close_on_complete: whether to close the mlflow run in `complete` phase in workflow, default to False.

    For more details of MLFlow usage, please refer to: https://mlflow.org/docs/latest/index.html.

    """

    # parameters that are logged at the start of training
    default_tracking_params = ["max_epochs", "epoch_length"]

    def __init__(
        self,
        tracking_uri: str | None = None,
        iteration_log: bool | Callable[[Engine, int], bool] = True,
        epoch_log: bool | Callable[[Engine, int], bool] = True,
        epoch_logger: Callable[[Engine], Any] | None = None,
        iteration_logger: Callable[[Engine], Any] | None = None,
        dataset_logger: Callable[[Mapping[str, Dataset]], Any] | None = None,
        dataset_dict: Mapping[str, Dataset] | None = None,
        dataset_keys: str = CommonKeys.IMAGE,
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
        self.dataset_logger = dataset_logger
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
        self.run_finish_status = mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.FINISHED)
        self.close_on_complete = close_on_complete
        self.experiment = None
        self.cur_run = None
        self.dataset_dict = dataset_dict
        self.dataset_keys = ensure_tuple(dataset_keys)

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
            event = Events.ITERATION_COMPLETED
            if callable(self.iteration_log):  # substitute event with new one using filter callable
                event = event(event_filter=self.iteration_log)
            engine.add_event_handler(event, self.iteration_completed)
        if self.epoch_log and not engine.has_event_handler(self.epoch_completed, Events.EPOCH_COMPLETED):
            event = Events.EPOCH_COMPLETED
            if callable(self.epoch_log):  # substitute event with new one using filter callable
                event = event(event_filter=self.epoch_log)
            engine.add_event_handler(event, self.epoch_completed)
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
            # runs marked as finish should not record info any more
            runs = [r for r in runs if r.info.status != self.run_finish_status]
            if runs:
                self.cur_run = self.client.get_run(runs[-1].info.run_id)  # pick latest active run
            else:
                self.cur_run = self.client.create_run(experiment_id=self.experiment.experiment_id, run_name=run_name)

        if self.experiment_param:
            self._log_params(self.experiment_param)

        attrs = {attr: getattr(engine.state, attr, None) for attr in self.default_tracking_params}
        self._delete_exist_param_in_dict(attrs)
        self._log_params(attrs)

        if self.dataset_logger:
            self.dataset_logger(self.dataset_dict)
        else:
            self._default_dataset_log(self.dataset_dict)

    def _set_experiment(self):
        experiment = self.experiment
        if not experiment:
            for _retry_time in range(3):
                try:
                    experiment = self.client.get_experiment_by_name(self.experiment_name)
                    if not experiment:
                        experiment_id = self.client.create_experiment(self.experiment_name)
                        experiment = self.client.get_experiment(experiment_id)
                    break
                except MlflowException as e:
                    if "RESOURCE_ALREADY_EXISTS" in str(e):
                        logger.warning("Experiment already exists; delaying before retrying.")
                        time.sleep(1)
                        if _retry_time == 2:
                            raise e
                    else:
                        raise e

        if experiment.lifecycle_stage != mlflow.entities.LifecycleStage.ACTIVE:
            raise ValueError(f"Cannot set a deleted experiment '{self.experiment_name}' as the active experiment")
        self.experiment = experiment

    @staticmethod
    def _get_pandas_dataset_info(pandas_dataset):
        dataset_name = pandas_dataset.name
        return {
            f"{dataset_name}_digest": pandas_dataset.digest,
            f"{dataset_name}_samples": pandas_dataset.profile["num_rows"],
        }

    def _log_dataset(self, sample_dict: dict[str, Any], context: str = "train") -> None:
        if not self.cur_run:
            raise ValueError("Current Run is not Active to log the dataset")

        # Need to update the self.cur_run to sync the dataset log, otherwise the `inputs` info will be out-of-date.
        self.cur_run = self.client.get_run(self.cur_run.info.run_id)
        logged_set = [x for x in self.cur_run.inputs.dataset_inputs if x.dataset.name.startswith(context)]
        # In case there are datasets with the same name.
        dataset_count = str(len(logged_set))
        dataset_name = f"{context}_dataset_{dataset_count}"
        sample_df = pandas.DataFrame(sample_dict)
        dataset = mlflow.data.from_pandas(sample_df, name=dataset_name)
        exist_dataset_list = list(
            filter(lambda x: x.dataset.digest == dataset.digest, self.cur_run.inputs.dataset_inputs)
        )

        if not len(exist_dataset_list):
            datasets = [mlflow.entities.DatasetInput(dataset._to_mlflow_entity())]
            self.client.log_inputs(run_id=self.cur_run.info.run_id, datasets=datasets)
            dataset_info = MLFlowHandler._get_pandas_dataset_info(dataset)
            self._log_params(dataset_info)

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
        metrics_arr = [
            mlflow.entities.Metric(key, value, timestamp, step or 0) for key, value in flatten_dict(metrics).items()
        ]
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
            self.client.set_terminated(self.cur_run.info.run_id, self.run_finish_status)
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
                    f"{param_name}_group_{i}": float(param_group[param_name])
                    for i, param_group in enumerate(cur_optimizer.param_groups)
                }
                self._log_metrics(params, step=engine.state.iteration)

    def _default_dataset_log(self, dataset_dict: Mapping[str, Dataset] | None) -> None:
        """
        Execute dataset log operation based on the input dataset_dict. The dataset_dict should have a format
        like:
            {
                "dataset_name0": dataset0,
                "dataset_name1": dataset1,
                ......
            }
        The keys stand for names of datasets, which will be logged as prefixes of dataset names in MLFlow.
        The values are PyTorch datasets from which sample names are abstracted to build a Pandas DataFrame.
        If the input dataset_dict is None, this function will directly return and do nothing.

        To use this function, every sample in the input datasets must contain keys specified by the `dataset_keys`
        parameter.
        This function will log a PandasDataset to MLFlow inputs, generated from the Pandas DataFrame.
        For more details about PandasDataset, please refer to this link:
        https://mlflow.org/docs/latest/python_api/mlflow.data.html#mlflow.data.pandas_dataset.PandasDataset

        Please note that it may take a while to record the dataset if it has too many samples.

        Args:
            dataset_dict: a dictionary in which the key is the name of the dataset and the value is a PyTorch
                dataset, that needs to be recorded.

        """

        if dataset_dict is None:
            return
        elif len(dataset_dict) == 0:
            warnings.warn("There is no dataset to log!")

        # Log datasets to MLFlow one by one.
        for dataset_type, dataset in dataset_dict.items():
            if dataset is None:
                raise AttributeError(f"The {dataset_type} dataset of is None. Cannot record it by MLFlow.")

            sample_dict: dict[str, list[str]] = {}
            dataset_samples = getattr(dataset, "data", [])
            for sample in tqdm(dataset_samples, f"Recording the {dataset_type} dataset"):
                for key in self.dataset_keys:
                    if key not in sample_dict:
                        sample_dict[key] = []

                    if key in sample:
                        value_to_log = sample[key]
                    else:
                        raise KeyError(f"Unexpect key '{key}' in the sample.")

                    if not isinstance(value_to_log, str):
                        warnings.warn(
                            f"Expected type string, got type {type(value_to_log)} of the {key} name."
                            "May log an empty dataset in MLFlow"
                        )
                    else:
                        sample_dict[key].append(value_to_log)
            self._log_dataset(sample_dict, dataset_type)

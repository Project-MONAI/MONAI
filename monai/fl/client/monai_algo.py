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
from collections.abc import Mapping, MutableMapping
from typing import Any, cast

import torch
import torch.distributed as dist

from monai.apps.auto3dseg.data_analyzer import DataAnalyzer
from monai.apps.utils import get_logger
from monai.auto3dseg import SegSummarizer
from monai.bundle import BundleWorkflow, ConfigComponent, ConfigItem, ConfigParser, ConfigWorkflow
from monai.engines import SupervisedEvaluator, SupervisedTrainer, Trainer
from monai.fl.client import ClientAlgo, ClientAlgoStats
from monai.fl.utils.constants import ExtraItems, FiltersType, FlPhase, FlStatistics, ModelType, WeightType
from monai.fl.utils.exchange_object import ExchangeObject
from monai.networks.utils import copy_model_state, get_state_dict
from monai.utils import min_version, require_pkg
from monai.utils.enums import DataStatsKeys

logger = get_logger(__name__)


def convert_global_weights(global_weights: Mapping, local_var_dict: MutableMapping) -> tuple[MutableMapping, int]:
    """Helper function to convert global weights to local weights format"""
    # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
    model_keys = global_weights.keys()
    n_converted = 0
    for var_name in local_var_dict:
        if var_name in model_keys:
            weights = global_weights[var_name]
            try:
                # reshape global weights to compute difference later on
                weights = torch.reshape(torch.as_tensor(weights), local_var_dict[var_name].shape)
                # update the local dict
                local_var_dict[var_name] = weights
                n_converted += 1
            except Exception as e:
                raise ValueError(f"Convert weight from {var_name} failed.") from e
    return local_var_dict, n_converted


def compute_weight_diff(global_weights, local_var_dict):
    if global_weights is None:
        raise ValueError("Cannot compute weight differences if `global_weights` is None!")
    if local_var_dict is None:
        raise ValueError("Cannot compute weight differences if `local_var_dict` is None!")
    # compute delta model, global model has the primary key set
    weight_diff = {}
    n_diff = 0
    for name in global_weights:
        if name not in local_var_dict:
            continue
        # returned weight diff will be on the cpu
        weight_diff[name] = local_var_dict[name].cpu() - global_weights[name].cpu()
        n_diff += 1
        if torch.any(torch.isnan(weight_diff[name])):
            raise ValueError(f"Weights for {name} became NaN...")
    if n_diff == 0:
        raise RuntimeError("No weight differences computed!")
    return weight_diff


def disable_ckpt_loaders(parser: ConfigParser) -> None:
    if "validate#handlers" in parser:
        for h in parser["validate#handlers"]:
            if ConfigComponent.is_instantiable(h):
                if "CheckpointLoader" in h["_target_"]:
                    h["_disabled_"] = True


class MonaiAlgoStats(ClientAlgoStats):
    """
    Implementation of ``ClientAlgoStats`` to allow federated learning with MONAI bundle configurations.

    Args:
        bundle_root: directory path of the bundle.
        config_train_filename: bundle training config path relative to bundle_root. Can be a list of files;
            defaults to "configs/train.json". only useful when `workflow` is None.
        config_filters_filename: filter configuration file. Can be a list of files; defaults to `None`.
        data_stats_transform_list: transforms to apply for the data stats result.
        histogram_only: whether to only compute histograms. Defaults to False.
        workflow: the bundle workflow to execute, usually it's training, evaluation or inference.
            if None, will create an `ConfigWorkflow` internally based on `config_train_filename`.
    """

    def __init__(
        self,
        bundle_root: str,
        config_train_filename: str | list | None = "configs/train.json",
        config_filters_filename: str | list | None = None,
        data_stats_transform_list: list | None = None,
        histogram_only: bool = False,
        workflow: BundleWorkflow | None = None,
    ):
        self.logger = logger
        self.bundle_root = bundle_root
        self.config_train_filename = config_train_filename
        self.config_filters_filename = config_filters_filename
        self.train_data_key = "train"
        self.eval_data_key = "eval"
        self.data_stats_transform_list = data_stats_transform_list
        self.histogram_only = histogram_only
        self.workflow = None
        if workflow is not None:
            if not isinstance(workflow, BundleWorkflow):
                raise ValueError("workflow must be a subclass of BundleWorkflow.")
            if workflow.get_workflow_type() is None:
                raise ValueError("workflow doesn't specify the type.")
            self.workflow = workflow

        self.client_name: str | None = None
        self.app_root: str = ""
        self.post_statistics_filters: Any = None
        self.phase = FlPhase.IDLE
        self.dataset_root: Any = None

    def initialize(self, extra=None):
        """
        Initialize routine to parse configuration files and extract main components such as trainer, evaluator, and filters.

        Args:
            extra: Dict with additional information that should be provided by FL system,
                i.e., `ExtraItems.CLIENT_NAME`, `ExtraItems.APP_ROOT` and `ExtraItems.LOGGING_FILE`.
                You can diable the logging logic in the monai bundle by setting {ExtraItems.LOGGING_FILE} to False.

        """
        if extra is None:
            extra = {}
        self.client_name = extra.get(ExtraItems.CLIENT_NAME, "noname")
        logging_file = extra.get(ExtraItems.LOGGING_FILE, None)
        self.logger.info(f"Initializing {self.client_name} ...")

        # FL platform needs to provide filepath to configuration files
        self.app_root = extra.get(ExtraItems.APP_ROOT, "")
        self.bundle_root = os.path.join(self.app_root, self.bundle_root)

        if self.workflow is None:
            config_train_files = self._add_config_files(self.config_train_filename)
            self.workflow = ConfigWorkflow(
                config_file=config_train_files, meta_file=None, logging_file=logging_file, workflow_type="train"
            )
        self.workflow.initialize()
        self.workflow.bundle_root = self.bundle_root
        # initialize the workflow as the content changed
        self.workflow.initialize()

        config_filter_files = self._add_config_files(self.config_filters_filename)
        filter_parser = ConfigParser()
        if len(config_filter_files) > 0:
            filter_parser.read_config(config_filter_files)
            # Get filters
            self.post_statistics_filters = filter_parser.get_parsed_content(
                FiltersType.POST_STATISTICS_FILTERS, default=ConfigItem(None, FiltersType.POST_STATISTICS_FILTERS)
            )
        self.logger.info(f"Initialized {self.client_name}.")

    def get_data_stats(self, extra: dict | None = None) -> ExchangeObject:
        """
        Returns summary statistics about the local data.

        Args:
            extra: Dict with additional information that can be provided by the FL system.
                    Both FlStatistics.HIST_BINS and FlStatistics.HIST_RANGE must be provided.

        Returns:
            stats: ExchangeObject with summary statistics.

        """
        if extra is None:
            raise ValueError("`extra` has to be set")

        if self.workflow.dataset_dir:  # type: ignore
            self.phase = FlPhase.GET_DATA_STATS
            self.logger.info(f"Computing statistics on {self.workflow.dataset_dir}")  # type: ignore

            if FlStatistics.HIST_BINS not in extra:
                raise ValueError("FlStatistics.NUM_OF_BINS not specified in `extra`")
            else:
                hist_bins = extra[FlStatistics.HIST_BINS]
            if FlStatistics.HIST_RANGE not in extra:
                raise ValueError("FlStatistics.HIST_RANGE not specified in `extra`")
            else:
                hist_range = extra[FlStatistics.HIST_RANGE]

            stats_dict = {}

            # train data stats
            train_summary_stats, train_case_stats = self._get_data_key_stats(
                data=self.workflow.train_dataset_data,  # type: ignore
                data_key=self.train_data_key,
                hist_bins=hist_bins,
                hist_range=hist_range,
                output_path=os.path.join(self.app_root, "train_data_stats.yaml"),
            )
            if train_case_stats:
                # Only return summary statistics to FL server
                stats_dict.update({self.train_data_key: train_summary_stats})

            # eval data stats
            eval_summary_stats = None
            eval_case_stats = None
            if self.workflow.val_dataset_data is not None:  # type: ignore
                eval_summary_stats, eval_case_stats = self._get_data_key_stats(
                    data=self.workflow.val_dataset_data,  # type: ignore
                    data_key=self.eval_data_key,
                    hist_bins=hist_bins,
                    hist_range=hist_range,
                    output_path=os.path.join(self.app_root, "eval_data_stats.yaml"),
                )
            else:
                self.logger.warning("the datalist doesn't contain validation section.")
            if eval_summary_stats:
                # Only return summary statistics to FL server
                stats_dict.update({self.eval_data_key: eval_summary_stats})

            # total stats
            if train_case_stats and eval_case_stats:
                # Compute total summary
                total_summary_stats = self._compute_total_stats(
                    [train_case_stats, eval_case_stats], hist_bins, hist_range
                )
                stats_dict.update({FlStatistics.TOTAL_DATA: total_summary_stats})

            # optional filter of data stats
            stats = ExchangeObject(statistics=stats_dict)
            if self.post_statistics_filters is not None:
                for _filter in self.post_statistics_filters:
                    stats = _filter(stats, extra)

            return stats
        else:
            raise ValueError("data_root not set!")

    def _get_data_key_stats(self, data, data_key, hist_bins, hist_range, output_path=None):
        analyzer = DataAnalyzer(
            datalist={data_key: data},
            dataroot=self.workflow.dataset_dir,  # type: ignore
            hist_bins=hist_bins,
            hist_range=hist_range,
            output_path=output_path,
            histogram_only=self.histogram_only,
        )

        self.logger.info(f"{self.client_name} compute data statistics on {data_key}...")
        all_stats = analyzer.get_all_case_stats(transform_list=self.data_stats_transform_list, key=data_key)

        case_stats = all_stats[DataStatsKeys.BY_CASE]

        summary_stats = {
            FlStatistics.DATA_STATS: all_stats[DataStatsKeys.SUMMARY],
            FlStatistics.DATA_COUNT: len(data),
            FlStatistics.FAIL_COUNT: len(data) - len(case_stats),
            # TODO: add shapes, voxels sizes, etc.
        }

        return summary_stats, case_stats

    @staticmethod
    def _compute_total_stats(case_stats_lists, hist_bins, hist_range):
        # Compute total summary
        total_case_stats = []
        for case_stats_list in case_stats_lists:
            total_case_stats += case_stats_list

        summarizer = SegSummarizer(
            "image", "label", average=True, do_ccp=True, hist_bins=hist_bins, hist_range=hist_range
        )
        total_summary_stats = summarizer.summarize(total_case_stats)

        summary_stats = {
            FlStatistics.DATA_STATS: total_summary_stats,
            FlStatistics.DATA_COUNT: len(total_case_stats),
            FlStatistics.FAIL_COUNT: 0,
        }

        return summary_stats

    def _add_config_files(self, config_files):
        files = []
        if config_files:
            if isinstance(config_files, str):
                files.append(os.path.join(self.bundle_root, config_files))
            elif isinstance(config_files, list):
                for file in config_files:
                    if isinstance(file, str):
                        files.append(os.path.join(self.bundle_root, file))
                    else:
                        raise ValueError(f"Expected config file to be of type str but got {type(file)}: {file}")
            else:
                raise ValueError(
                    f"Expected config files to be of type str or list but got {type(config_files)}: {config_files}"
                )
        return files


@require_pkg(pkg_name="ignite", version="0.4.10", version_checker=min_version)
class MonaiAlgo(ClientAlgo, MonaiAlgoStats):
    """
    Implementation of ``ClientAlgo`` to allow federated learning with MONAI bundle configurations.

    Args:
        bundle_root: directory path of the bundle.
        local_epochs: number of local epochs to execute during each round of local training; defaults to 1.
        send_weight_diff: whether to send weight differences rather than full weights; defaults to `True`.
        config_train_filename: bundle training config path relative to bundle_root. can be a list of files.
            defaults to "configs/train.json". only useful when `train_workflow` is None.
        train_kwargs: other args of the `ConfigWorkflow` of train, except for `config_file`, `meta_file`,
            `logging_file`, `workflow_type`. only useful when `train_workflow` is None.
        config_evaluate_filename: bundle evaluation config path relative to bundle_root. can be a list of files.
            if "default", ["configs/train.json", "configs/evaluate.json"] will be used.
            this arg is only useful when `eval_workflow` is None.
        eval_kwargs: other args of the `ConfigWorkflow` of evaluation, except for `config_file`, `meta_file`,
            `logging_file`, `workflow_type`. only useful when `eval_workflow` is None.
        config_filters_filename: filter configuration file. Can be a list of files; defaults to `None`.
        disable_ckpt_loading: do not use any CheckpointLoader if defined in train/evaluate configs; defaults to `True`.
        best_model_filepath: location of best model checkpoint; defaults "models/model.pt" relative to `bundle_root`.
        final_model_filepath: location of final model checkpoint; defaults "models/model_final.pt" relative to `bundle_root`.
        save_dict_key: If a model checkpoint contains several state dicts,
            the one defined by `save_dict_key` will be returned by `get_weights`; defaults to "model".
            If all state dicts should be returned, set `save_dict_key` to None.
        data_stats_transform_list: transforms to apply for the data stats result.
        eval_workflow_name: the workflow name corresponding to the "config_evaluate_filename", default to "train"
            as the default "config_evaluate_filename" overrides the train workflow config.
            this arg is only useful when `eval_workflow` is None.
        train_workflow: the bundle workflow to execute training, if None, will create a `ConfigWorkflow` internally
            based on `config_train_filename` and `train_kwargs`.
        eval_workflow: the bundle workflow to execute evaluation, if None, will create a `ConfigWorkflow` internally
            based on `config_evaluate_filename`, `eval_kwargs`, `eval_workflow_name`.

    """

    def __init__(
        self,
        bundle_root: str,
        local_epochs: int = 1,
        send_weight_diff: bool = True,
        config_train_filename: str | list | None = "configs/train.json",
        train_kwargs: dict | None = None,
        config_evaluate_filename: str | list | None = "default",
        eval_kwargs: dict | None = None,
        config_filters_filename: str | list | None = None,
        disable_ckpt_loading: bool = True,
        best_model_filepath: str | None = "models/model.pt",
        final_model_filepath: str | None = "models/model_final.pt",
        save_dict_key: str | None = "model",
        data_stats_transform_list: list | None = None,
        eval_workflow_name: str = "train",
        train_workflow: BundleWorkflow | None = None,
        eval_workflow: BundleWorkflow | None = None,
    ):
        self.logger = logger
        self.bundle_root = bundle_root
        self.local_epochs = local_epochs
        self.send_weight_diff = send_weight_diff
        self.config_train_filename = config_train_filename
        self.train_kwargs = {} if train_kwargs is None else train_kwargs
        if config_evaluate_filename == "default":
            # by default, evaluator needs both training and evaluate to be instantiated
            config_evaluate_filename = ["configs/train.json", "configs/evaluate.json"]
        self.config_evaluate_filename = config_evaluate_filename
        self.eval_kwargs = {} if eval_kwargs is None else eval_kwargs
        self.config_filters_filename = config_filters_filename
        self.disable_ckpt_loading = disable_ckpt_loading
        self.model_filepaths = {ModelType.BEST_MODEL: best_model_filepath, ModelType.FINAL_MODEL: final_model_filepath}
        self.save_dict_key = save_dict_key
        self.data_stats_transform_list = data_stats_transform_list
        self.eval_workflow_name = eval_workflow_name
        self.train_workflow = None
        self.eval_workflow = None
        if train_workflow is not None:
            if not isinstance(train_workflow, BundleWorkflow) or train_workflow.get_workflow_type() != "train":
                raise ValueError(
                    f"train workflow must be BundleWorkflow and set type in {BundleWorkflow.supported_train_type}."
                )
            self.train_workflow = train_workflow
        if eval_workflow is not None:
            # evaluation workflow can be "train" type or "infer" type
            if not isinstance(eval_workflow, BundleWorkflow) or eval_workflow.get_workflow_type() is None:
                raise ValueError("train workflow must be BundleWorkflow and set type.")
            self.eval_workflow = eval_workflow
        self.stats_sender = None

        self.app_root = ""
        self.filter_parser: ConfigParser | None = None
        self.trainer: SupervisedTrainer | None = None
        self.evaluator: SupervisedEvaluator | None = None
        self.pre_filters = None
        self.post_weight_filters = None
        self.post_evaluate_filters = None
        self.iter_of_start_time = 0
        self.global_weights: Mapping | None = None

        self.phase = FlPhase.IDLE
        self.client_name = None
        self.dataset_root = None

    def initialize(self, extra=None):
        """
        Initialize routine to parse configuration files and extract main components such as trainer, evaluator, and filters.

        Args:
            extra: Dict with additional information that should be provided by FL system,
                i.e., `ExtraItems.CLIENT_NAME`, `ExtraItems.APP_ROOT` and `ExtraItems.LOGGING_FILE`.
                You can diable the logging logic in the monai bundle by setting {ExtraItems.LOGGING_FILE} to False.

        """
        self._set_cuda_device()
        if extra is None:
            extra = {}
        self.client_name = extra.get(ExtraItems.CLIENT_NAME, "noname")
        logging_file = extra.get(ExtraItems.LOGGING_FILE, None)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"Initializing {self.client_name} ...")
        # FL platform needs to provide filepath to configuration files
        self.app_root = extra.get(ExtraItems.APP_ROOT, "")
        self.bundle_root = os.path.join(self.app_root, self.bundle_root)

        if self.train_workflow is None and self.config_train_filename is not None:
            config_train_files = self._add_config_files(self.config_train_filename)
            # if enabled experiment tracking, set the run name to the FL client name and timestamp,
            # expect the tracking settings use "run_name" to define the run name
            if "run_name" not in self.train_kwargs:
                self.train_kwargs["run_name"] = f"{self.client_name}_{timestamp}"
            self.train_workflow = ConfigWorkflow(
                config_file=config_train_files,
                meta_file=None,
                logging_file=logging_file,
                workflow_type="train",
                **self.train_kwargs,
            )
        if self.train_workflow is not None:
            self.train_workflow.initialize()
            self.train_workflow.bundle_root = self.bundle_root
            self.train_workflow.max_epochs = self.local_epochs
            if self.disable_ckpt_loading and isinstance(self.train_workflow, ConfigWorkflow):
                disable_ckpt_loaders(parser=self.train_workflow.parser)
            # initialize the workflow as the content changed
            self.train_workflow.initialize()
            self.trainer = self.train_workflow.trainer
            if not isinstance(self.trainer, SupervisedTrainer):
                raise ValueError(f"trainer must be SupervisedTrainer, but got: {type(self.trainer)}.")

        if self.eval_workflow is None and self.config_evaluate_filename is not None:
            config_eval_files = self._add_config_files(self.config_evaluate_filename)
            # if enabled experiment tracking, set the run name to the FL client name and timestamp,
            # expect the tracking settings use "run_name" to define the run name
            if "run_name" not in self.eval_kwargs:
                self.eval_kwargs["run_name"] = f"{self.client_name}_{timestamp}"
            self.eval_workflow = ConfigWorkflow(
                config_file=config_eval_files,
                meta_file=None,
                logging_file=logging_file,
                workflow_type=self.eval_workflow_name,
                **self.eval_kwargs,
            )
        if self.eval_workflow is not None:
            self.eval_workflow.initialize()
            self.eval_workflow.bundle_root = self.bundle_root
            if self.disable_ckpt_loading and isinstance(self.eval_workflow, ConfigWorkflow):
                disable_ckpt_loaders(parser=self.eval_workflow.parser)
            # initialize the workflow as the content changed
            self.eval_workflow.initialize()
            self.evaluator = self.eval_workflow.evaluator
            if not isinstance(self.evaluator, SupervisedEvaluator):
                raise ValueError(f"evaluator must be SupervisedEvaluator, but got: {type(self.evaluator)}.")

        config_filter_files = self._add_config_files(self.config_filters_filename)
        self.filter_parser = ConfigParser()
        if len(config_filter_files) > 0:
            self.filter_parser.read_config(config_filter_files)

        # set stats sender for nvflare
        self.stats_sender = extra.get(ExtraItems.STATS_SENDER, self.stats_sender)
        if self.stats_sender is not None:
            self.stats_sender.attach(self.trainer)
            self.stats_sender.attach(self.evaluator)

        # Get filters
        self.pre_filters = self.filter_parser.get_parsed_content(
            FiltersType.PRE_FILTERS, default=ConfigItem(None, FiltersType.PRE_FILTERS)
        )
        self.post_weight_filters = self.filter_parser.get_parsed_content(
            FiltersType.POST_WEIGHT_FILTERS, default=ConfigItem(None, FiltersType.POST_WEIGHT_FILTERS)
        )
        self.post_evaluate_filters = self.filter_parser.get_parsed_content(
            FiltersType.POST_EVALUATE_FILTERS, default=ConfigItem(None, FiltersType.POST_EVALUATE_FILTERS)
        )
        self.post_statistics_filters = self.filter_parser.get_parsed_content(
            FiltersType.POST_STATISTICS_FILTERS, default=ConfigItem(None, FiltersType.POST_STATISTICS_FILTERS)
        )
        self.logger.info(f"Initialized {self.client_name}.")

    def train(self, data: ExchangeObject, extra: dict | None = None) -> None:
        """
        Train on client's local data.

        Args:
            data: `ExchangeObject` containing the current global model weights.
            extra: Dict with additional information that can be provided by the FL system.

        """

        self._set_cuda_device()
        if extra is None:
            extra = {}
        if not isinstance(data, ExchangeObject):
            raise ValueError(f"expected data to be ExchangeObject but received {type(data)}")

        if self.trainer is None:
            raise ValueError("self.trainer should not be None.")
        if self.pre_filters is not None:
            for _filter in self.pre_filters:
                data = _filter(data, extra)
        self.phase = FlPhase.TRAIN
        self.logger.info(f"Load {self.client_name} weights...")
        local_var_dict = get_state_dict(self.trainer.network)
        self.global_weights, n_converted = convert_global_weights(
            global_weights=cast(dict, data.weights), local_var_dict=local_var_dict
        )
        self._check_converted(data.weights, local_var_dict, n_converted)

        # set engine state max epochs.
        self.trainer.state.max_epochs = self.trainer.state.epoch + self.local_epochs
        # get current iteration when a round starts
        self.iter_of_start_time = self.trainer.state.iteration

        _, updated_keys, _ = copy_model_state(src=cast(Mapping, self.global_weights), dst=self.trainer.network)
        if len(updated_keys) == 0:
            self.logger.warning("No weights loaded!")
        self.logger.info(f"Start {self.client_name} training...")
        self.trainer.run()

    def get_weights(self, extra=None):
        """
        Returns the current weights of the model.

        Args:
            extra: Dict with additional information that can be provided by the FL system.

        Returns:
            return_weights: `ExchangeObject` containing current weights (default)
                or load requested model type from disk (`ModelType.BEST_MODEL` or `ModelType.FINAL_MODEL`).

        """

        self._set_cuda_device()
        if extra is None:
            extra = {}

        # by default return current weights, return best if requested via model type.
        self.phase = FlPhase.GET_WEIGHTS

        if ExtraItems.MODEL_TYPE in extra:
            model_type = extra.get(ExtraItems.MODEL_TYPE)
            if not isinstance(model_type, ModelType):
                raise ValueError(
                    f"Expected requested model type to be of type `ModelType` but received {type(model_type)}"
                )
            if model_type in self.model_filepaths:
                model_path = os.path.join(self.bundle_root, cast(str, self.model_filepaths[model_type]))
                if not os.path.isfile(model_path):
                    raise ValueError(f"No best model checkpoint exists at {model_path}")
                weights = torch.load(model_path, map_location="cpu", weights_only=True)
                # if weights contain several state dicts, use the one defined by `save_dict_key`
                if isinstance(weights, dict) and self.save_dict_key in weights:
                    weights = weights.get(self.save_dict_key)
                weigh_type: WeightType | None = WeightType.WEIGHTS
                stats: dict = {}
                self.logger.info(f"Returning {model_type} checkpoint weights from {model_path}.")
            else:
                raise ValueError(
                    f"Requested model type {model_type} not specified in `model_filepaths`: {self.model_filepaths}"
                )
        else:
            if self.trainer:
                weights = get_state_dict(self.trainer.network)
                # returned weights will be on the cpu
                for k in weights.keys():
                    weights[k] = weights[k].cpu()
                weigh_type = WeightType.WEIGHTS
                stats = self.trainer.get_stats()
                # calculate current iteration and epoch data after training.
                stats[FlStatistics.NUM_EXECUTED_ITERATIONS] = self.trainer.state.iteration - self.iter_of_start_time
                # compute weight differences
                if self.send_weight_diff:
                    weights = compute_weight_diff(global_weights=self.global_weights, local_var_dict=weights)
                    weigh_type = WeightType.WEIGHT_DIFF
                    self.logger.info("Returning current weight differences.")
                else:
                    self.logger.info("Returning current weights.")
            else:
                weights = None
                weigh_type = None
                stats = dict()

        if not isinstance(stats, dict):
            raise ValueError(f"stats is not a dict, {stats}")
        return_weights = ExchangeObject(
            weights=weights,
            optim=None,  # could be self.optimizer.state_dict()
            weight_type=weigh_type,
            statistics=stats,
        )

        # filter weights if needed (use to apply differential privacy, encryption, compression, etc.)
        if self.post_weight_filters is not None:
            for _filter in self.post_weight_filters:
                return_weights = _filter(return_weights, extra)

        return return_weights

    def evaluate(self, data: ExchangeObject, extra: dict | None = None) -> ExchangeObject:
        """
        Evaluate on client's local data.

        Args:
            data: `ExchangeObject` containing the current global model weights.
            extra: Dict with additional information that can be provided by the FL system.

        Returns:
            return_metrics: `ExchangeObject` containing evaluation metrics.

        """

        self._set_cuda_device()
        if extra is None:
            extra = {}
        if not isinstance(data, ExchangeObject):
            raise ValueError(f"expected data to be ExchangeObject but received {type(data)}")

        if self.evaluator is None:
            raise ValueError("self.evaluator should not be None.")
        if self.pre_filters is not None:
            for _filter in self.pre_filters:
                data = _filter(data, extra)

        self.phase = FlPhase.EVALUATE
        self.logger.info(f"Load {self.client_name} weights...")
        local_var_dict = get_state_dict(self.evaluator.network)
        global_weights, n_converted = convert_global_weights(
            global_weights=cast(dict, data.weights), local_var_dict=local_var_dict
        )
        self._check_converted(data.weights, local_var_dict, n_converted)

        _, updated_keys, _ = copy_model_state(src=global_weights, dst=self.evaluator.network)
        if len(updated_keys) == 0:
            self.logger.warning("No weights loaded!")
        self.logger.info(f"Start {self.client_name} evaluating...")
        if isinstance(self.trainer, Trainer):
            self.evaluator.run(self.trainer.state.epoch + 1)
        else:
            self.evaluator.run()
        return_metrics = ExchangeObject(metrics=self.evaluator.state.metrics)

        if self.post_evaluate_filters is not None:
            for _filter in self.post_evaluate_filters:
                return_metrics = _filter(return_metrics, extra)
        return return_metrics

    def abort(self, extra=None):
        """
        Abort the training or evaluation.
        Args:
            extra: Dict with additional information that can be provided by the FL system.
        """
        self.logger.info(f"Aborting {self.client_name} during {self.phase} phase.")
        if isinstance(self.trainer, Trainer):
            self.logger.info(f"Aborting {self.client_name} trainer...")
            self.trainer.interrupt()
        if isinstance(self.evaluator, Trainer):
            self.logger.info(f"Aborting {self.client_name} evaluator...")
            self.evaluator.interrupt()

    def finalize(self, extra: dict | None = None) -> None:
        """
        Finalize the training or evaluation.
        Args:
            extra: Dict with additional information that can be provided by the FL system.
        """
        self.logger.info(f"Terminating {self.client_name} during {self.phase} phase.")
        if isinstance(self.trainer, Trainer):
            self.logger.info(f"Terminating {self.client_name} trainer...")
            self.trainer.terminate()
        if isinstance(self.evaluator, Trainer):
            self.logger.info(f"Terminating {self.client_name} evaluator...")
            self.evaluator.terminate()
        if self.train_workflow is not None:
            self.train_workflow.finalize()
        if self.eval_workflow is not None:
            self.eval_workflow.finalize()

    def _check_converted(self, global_weights, local_var_dict, n_converted):
        if n_converted == 0:
            raise RuntimeError(
                f"No global weights converted! Received weight dict keys are {list(global_weights.keys())}"
            )
        else:
            self.logger.info(
                f"Converted {n_converted} global variables to match {len(local_var_dict)} local variables."
            )

    def _set_cuda_device(self):
        if dist.is_initialized():
            self.rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.rank)

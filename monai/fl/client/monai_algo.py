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

import logging
import os
import sys
from typing import Optional, Union

import torch
import torch.distributed as dist

import monai
from monai.apps.auto3dseg.data_analyzer import DataAnalyzer
from monai.auto3dseg import SegSummarizer
from monai.bundle import ConfigParser
from monai.bundle.config_item import ConfigComponent, ConfigItem
from monai.fl.client import ClientAlgo, ClientAlgoStats
from monai.fl.utils.constants import (
    BundleKeys,
    ExtraItems,
    FiltersType,
    FlPhase,
    FlStatistics,
    ModelType,
    RequiredBundleKeys,
    WeightType,
)
from monai.fl.utils.exchange_object import ExchangeObject
from monai.networks.utils import copy_model_state, get_state_dict
from monai.utils import min_version, require_pkg
from monai.utils.enums import DataStatsKeys, ImageStatsKeys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")


def convert_global_weights(global_weights, local_var_dict):
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
    for name in global_weights:
        if name not in local_var_dict:
            continue
        # returned weight diff will be on the cpu
        weight_diff[name] = local_var_dict[name].cpu() - global_weights[name].cpu()
        if torch.any(torch.isnan(weight_diff[name])):
            raise ValueError(f"Weights for {name} became NaN...")
    return weight_diff


def check_bundle_config(parser):
    for k in RequiredBundleKeys:
        if parser.get(k, None) is None:
            raise KeyError(f"Bundle config misses required key `{k}`")


def disable_ckpt_loaders(parser):
    if BundleKeys.VALIDATE_HANDLERS in parser:
        for h in parser[BundleKeys.VALIDATE_HANDLERS]:
            if ConfigComponent.is_instantiable(h):
                if "CheckpointLoader" in h["_target_"]:
                    h["_disabled_"] = True


class MonaiAlgoStats(ClientAlgoStats):
    """
    Implementation of ``ClientAlgo`` to allow federated learning with MONAI bundle configurations.

    Args:
        bundle_root: path of bundle.
        config_train_filename: bundle training config path relative to bundle_root. Can be a list of files;
            defaults to "configs/train.json".
        config_filters_filename: filter configuration file. Can be a list of files; defaults to `None`.
    """

    def __init__(
        self,
        bundle_root: str,
        config_train_filename: Optional[Union[str, list]] = "configs/train.json",
        config_filters_filename: Optional[Union[str, list]] = None,
        train_data_key: Optional[str] = BundleKeys.TRAIN_DATA,
        eval_data_key: Optional[str] = BundleKeys.VALID_DATA,
        data_stats_transform_list: Optional[list] = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.bundle_root = bundle_root
        self.config_train_filename = config_train_filename
        self.config_filters_filename = config_filters_filename
        self.train_data_key = train_data_key
        self.eval_data_key = eval_data_key
        self.data_stats_transform_list = data_stats_transform_list

        self.client_name = None
        self.app_root = None
        self.train_parser = None
        self.filter_parser = None
        self.post_statistics_filters = None
        self.phase = FlPhase.IDLE
        self.dataset_root = None

    def initialize(self, extra=None):
        """
        Initialize routine to parse configuration files and extract main components such as trainer, evaluator, and filters.

        Args:
            extra: Dict with additional information that should be provided by FL system,
                i.e., `ExtraItems.CLIENT_NAME` and `ExtraItems.APP_ROOT`.

        """
        if extra is None:
            extra = {}
        self.client_name = extra.get(ExtraItems.CLIENT_NAME, "noname")
        self.logger.info(f"Initializing {self.client_name} ...")

        # FL platform needs to provide filepath to configuration files
        self.app_root = extra.get(ExtraItems.APP_ROOT, "")

        # Read bundle config files
        self.bundle_root = os.path.join(self.app_root, self.bundle_root)

        config_train_files = self._add_config_files(self.config_train_filename)
        config_filter_files = self._add_config_files(self.config_filters_filename)

        # Parse
        self.train_parser = ConfigParser()
        self.filter_parser = ConfigParser()
        if len(config_train_files) > 0:
            self.train_parser.read_config(config_train_files)
            check_bundle_config(self.train_parser)
        if len(config_filter_files) > 0:
            self.filter_parser.read_config(config_filter_files)

        # override some config items
        self.train_parser[RequiredBundleKeys.BUNDLE_ROOT] = self.bundle_root

        # Get data location
        self.dataset_root = self.train_parser.get_parsed_content(
            BundleKeys.DATASET_DIR, default=ConfigItem(None, BundleKeys.DATASET_DIR)
        )

        # Get filters
        self.post_statistics_filters = self.filter_parser.get_parsed_content(
            FiltersType.POST_STATISTICS_FILTERS, default=ConfigItem(None, FiltersType.POST_STATISTICS_FILTERS)
        )

        self.logger.info(f"Initialized {self.client_name}.")

    def get_data_stats(self, extra: Optional[dict] = None) -> ExchangeObject:
        """
        Returns summary statistics about the local data.

        Args:
            extra: Dict with additional information that can be provided by the FL system.

        Returns:
            stats: ExchangeObject with summary statistics.

        """

        if self.dataset_root:
            self.phase = FlPhase.GET_DATA_STATS
            self.logger.info(f"Computing statistics on {self.dataset_root}")

            if FlStatistics.HIST_BINS not in extra:
                raise ValueError("FlStatistics.NUM_OF_BINS not specified in `extra`")
            else:
                hist_bins = extra[FlStatistics.HIST_BINS]
            if FlStatistics.HIST_RANGE not in extra:
                raise ValueError("FlStatistics.HIST_RANGE not specified in `extra`")
            else:
                hist_range = extra[FlStatistics.HIST_RANGE]

            train_summary_stats, train_case_stats = self._get_data_key_stats(
                parser=self.train_parser,
                data_key=self.train_data_key,
                hist_bins=hist_bins,
                hist_range=hist_range,
                output_path=os.path.join(self.app_root, "train_data_stats.yaml"),
            )
            eval_summary_stats, eval_case_stats = self._get_data_key_stats(
                parser=self.train_parser,
                data_key=self.eval_data_key,
                hist_bins=hist_bins,
                hist_range=hist_range,
                output_path=os.path.join(self.app_root, "eval_data_stats.yaml"),
            )
            # Compute total summary
            total_summary_stats = self._compute_total_stats([train_case_stats, eval_case_stats], hist_bins, hist_range)

            # Only return summary statistics to FL server
            stats = ExchangeObject(
                statistics={
                    self.train_data_key: train_summary_stats,
                    self.eval_data_key: eval_summary_stats,
                    FlStatistics.TOTAL_DATA: total_summary_stats,
                }
            )

            if self.post_statistics_filters is not None:
                for _filter in self.post_statistics_filters:
                    stats = _filter(stats, extra)

            return stats
        else:
            raise ValueError("data_root not set!")

    def _get_data_key_stats(self, parser, data_key, hist_bins, hist_range, output_path=None):
        data = parser.get_parsed_content(data_key)

        datalist = {data_key: data}

        analyzer = DataAnalyzer(
            datalist=datalist,
            dataroot=self.dataset_root,
            hist_bins=hist_bins,
            hist_range=hist_range,
            output_path=output_path,
        )

        self.logger.info(f"{self.client_name} compute data statistics on {data_key}...")
        all_stats = analyzer.get_all_case_stats(transform_list=self.data_stats_transform_list, key=data_key)

        case_stats = all_stats[DataStatsKeys.BY_CASE]
        nr_histograms = len(all_stats[DataStatsKeys.SUMMARY][DataStatsKeys.IMAGE_HISTOGRAM][ImageStatsKeys.HISTOGRAM])
        if nr_histograms > 1:
            feature_names = [f"Intensity-{i}" for i in range(nr_histograms)]
        else:
            feature_names = ["Intensity"]

        summary_stats = {
            FlStatistics.DATA_STATS: all_stats[DataStatsKeys.SUMMARY],
            FlStatistics.DATA_COUNT: len(data),
            FlStatistics.FAIL_COUNT: len(data)
            - len(case_stats),  # TODO: is this robust? Call a failure if any analyze fails?
            # TODO: add shapes, voxels sizes, etc.
            FlStatistics.FEATURE_NAMES: feature_names,
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

        nr_histograms = len(total_summary_stats[DataStatsKeys.IMAGE_HISTOGRAM][ImageStatsKeys.HISTOGRAM])
        if nr_histograms > 1:
            feature_names = [f"Intensity-{i}" for i in range(nr_histograms)]
        else:
            feature_names = ["Intensity"]

        summary_stats = {
            FlStatistics.DATA_STATS: total_summary_stats,
            FlStatistics.DATA_COUNT: len(total_case_stats),
            FlStatistics.FAIL_COUNT: 0,
            FlStatistics.FEATURE_NAMES: feature_names,
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
        bundle_root: path of bundle.
        local_epochs: number of local epochs to execute during each round of local training; defaults to 1.
        send_weight_diff: whether to send weight differences rather than full weights; defaults to `True`.
        config_train_filename: bundle training config path relative to bundle_root. Can be a list of files;
            defaults to "configs/train.json".
        config_evaluate_filename: bundle evaluation config path relative to bundle_root. Can be a list of files.
            If "default", config_evaluate_filename = ["configs/train.json", "configs/evaluate.json"] will be used;
        config_filters_filename: filter configuration file. Can be a list of files; defaults to `None`.
        disable_ckpt_loading: do not use any CheckpointLoader if defined in train/evaluate configs; defaults to `True`.
        best_model_filepath: location of best model checkpoint; defaults "models/model.pt" relative to `bundle_root`.
        final_model_filepath: location of final model checkpoint; defaults "models/model_final.pt" relative to `bundle_root`.
        save_dict_key: If a model checkpoint contains several state dicts,
            the one defined by `save_dict_key` will be returned by `get_weights`; defaults to "model".
            If all state dicts should be returned, set `save_dict_key` to None.
        seed: set random seed for modules to enable or disable deterministic training; defaults to `None`,
            i.e., non-deterministic training.
        benchmark: set benchmark to `False` for full deterministic behavior in cuDNN components.
            Note, full determinism in federated learning depends also on deterministic behavior of other FL components,
            e.g., the aggregator, which is not controlled by this class.
        multi_gpu: whether to run MonaiAlgo in a multi-GPU setting; defaults to `False`.
        backend: backend to use for torch.distributed; defaults to "nccl".
        init_method: init_method for torch.distributed; defaults to "env://".
    """

    def __init__(
        self,
        bundle_root: str,
        local_epochs: int = 1,
        send_weight_diff: bool = True,
        config_train_filename: Optional[Union[str, list]] = "configs/train.json",
        config_evaluate_filename: Optional[Union[str, list]] = "default",
        config_filters_filename: Optional[Union[str, list]] = None,
        disable_ckpt_loading: bool = True,
        best_model_filepath: Optional[str] = "models/model.pt",
        final_model_filepath: Optional[str] = "models/model_final.pt",
        save_dict_key: Optional[str] = "model",
        seed: Optional[int] = None,
        benchmark: bool = True,
        multi_gpu: bool = False,
        backend: str = "nccl",
        init_method: str = "env://",
        train_data_key: Optional[str] = BundleKeys.TRAIN_DATA,
        eval_data_key: Optional[str] = BundleKeys.VALID_DATA,
        data_stats_transform_list: Optional[list] = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        if config_evaluate_filename == "default":
            # by default, evaluator needs both training and evaluate to be instantiated.
            config_evaluate_filename = ["configs/train.json", "configs/evaluate.json"]
        self.bundle_root = bundle_root
        self.local_epochs = local_epochs
        self.send_weight_diff = send_weight_diff
        self.config_train_filename = config_train_filename
        self.config_evaluate_filename = config_evaluate_filename
        self.config_filters_filename = config_filters_filename
        self.disable_ckpt_loading = disable_ckpt_loading
        self.model_filepaths = {ModelType.BEST_MODEL: best_model_filepath, ModelType.FINAL_MODEL: final_model_filepath}
        self.save_dict_key = save_dict_key
        self.seed = seed
        self.benchmark = benchmark
        self.multi_gpu = multi_gpu
        self.backend = backend
        self.init_method = init_method
        self.train_data_key = train_data_key
        self.eval_data_key = eval_data_key
        self.data_stats_transform_list = data_stats_transform_list

        self.app_root = None
        self.train_parser = None
        self.eval_parser = None
        self.filter_parser = None
        self.trainer = None
        self.evaluator = None
        self.pre_filters = None
        self.post_weight_filters = None
        self.post_evaluate_filters = None
        self.iter_of_start_time = 0
        self.global_weights = None
        self.rank = 0

        self.phase = FlPhase.IDLE
        self.client_name = None
        self.dataset_root = None

    def initialize(self, extra=None):
        """
        Initialize routine to parse configuration files and extract main components such as trainer, evaluator, and filters.

        Args:
            extra: Dict with additional information that should be provided by FL system,
                i.e., `ExtraItems.CLIENT_NAME` and `ExtraItems.APP_ROOT`.

        """
        if extra is None:
            extra = {}
        self.client_name = extra.get(ExtraItems.CLIENT_NAME, "noname")
        self.logger.info(f"Initializing {self.client_name} ...")

        if self.multi_gpu:
            dist.init_process_group(backend=self.backend, init_method=self.init_method)
            self._set_cuda_device()
            self.logger.info(
                f"Using multi-gpu training on rank {self.rank} (available devices: {torch.cuda.device_count()})"
            )
            if self.rank > 0:
                self.logger.setLevel(logging.WARNING)

        if self.seed:
            monai.utils.set_determinism(seed=self.seed)
        torch.backends.cudnn.benchmark = self.benchmark

        # FL platform needs to provide filepath to configuration files
        self.app_root = extra.get(ExtraItems.APP_ROOT, "")

        # Read bundle config files
        self.bundle_root = os.path.join(self.app_root, self.bundle_root)

        config_train_files = self._add_config_files(self.config_train_filename)
        config_eval_files = self._add_config_files(self.config_evaluate_filename)
        config_filter_files = self._add_config_files(self.config_filters_filename)

        # Parse
        self.train_parser = ConfigParser()
        self.eval_parser = ConfigParser()
        self.filter_parser = ConfigParser()
        if len(config_train_files) > 0:
            self.train_parser.read_config(config_train_files)
            check_bundle_config(self.train_parser)
        if len(config_eval_files) > 0:
            self.eval_parser.read_config(config_eval_files)
            check_bundle_config(self.eval_parser)
        if len(config_filter_files) > 0:
            self.filter_parser.read_config(config_filter_files)

        # override some config items
        self.train_parser[RequiredBundleKeys.BUNDLE_ROOT] = self.bundle_root
        self.eval_parser[RequiredBundleKeys.BUNDLE_ROOT] = self.bundle_root
        # number of training epochs for each round
        if BundleKeys.TRAIN_TRAINER_MAX_EPOCHS in self.train_parser:
            self.train_parser[BundleKeys.TRAIN_TRAINER_MAX_EPOCHS] = self.local_epochs

        # remove checkpoint loaders
        if self.disable_ckpt_loading:
            disable_ckpt_loaders(self.train_parser)
            disable_ckpt_loaders(self.eval_parser)

        # Get trainer, evaluator
        self.trainer = self.train_parser.get_parsed_content(
            BundleKeys.TRAINER, default=ConfigItem(None, BundleKeys.TRAINER)
        )
        self.evaluator = self.eval_parser.get_parsed_content(
            BundleKeys.EVALUATOR, default=ConfigItem(None, BundleKeys.EVALUATOR)
        )

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

        # Get data location
        self.dataset_root = self.train_parser.get_parsed_content(
            BundleKeys.DATASET_DIR, default=ConfigItem(None, BundleKeys.DATASET_DIR)
        )

        self.logger.info(f"Initialized {self.client_name}.")

    def train(self, data: ExchangeObject, extra=None):
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
            global_weights=data.weights, local_var_dict=local_var_dict
        )
        self._check_converted(data.weights, local_var_dict, n_converted)

        # set engine state max epochs.
        self.trainer.state.max_epochs = self.trainer.state.epoch + self.local_epochs
        # get current iteration when a round starts
        self.iter_of_start_time = self.trainer.state.iteration

        _, updated_keys, _ = copy_model_state(src=self.global_weights, dst=self.trainer.network)
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
                model_path = os.path.join(self.bundle_root, self.model_filepaths[model_type])
                if not os.path.isfile(model_path):
                    raise ValueError(f"No best model checkpoint exists at {model_path}")
                weights = torch.load(model_path, map_location="cpu")
                # if weights contain several state dicts, use the one defined by `save_dict_key`
                if isinstance(weights, dict) and self.save_dict_key in weights:
                    weights = weights.get(self.save_dict_key)
                weigh_type = WeightType.WEIGHTS
                stats = dict()
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

    def evaluate(self, data: ExchangeObject, extra=None):
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
        global_weights, n_converted = convert_global_weights(global_weights=data.weights, local_var_dict=local_var_dict)
        self._check_converted(data.weights, local_var_dict, n_converted)

        _, updated_keys, _ = copy_model_state(src=global_weights, dst=self.evaluator.network)
        if len(updated_keys) == 0:
            self.logger.warning("No weights loaded!")
        self.logger.info(f"Start {self.client_name} evaluating...")
        if isinstance(self.trainer, monai.engines.Trainer):
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
        if isinstance(self.trainer, monai.engines.Trainer):
            self.logger.info(f"Aborting {self.client_name} trainer...")
            self.trainer.interrupt()
        if isinstance(self.evaluator, monai.engines.Trainer):
            self.logger.info(f"Aborting {self.client_name} evaluator...")
            self.evaluator.interrupt()

    def finalize(self, extra=None):
        """
        Finalize the training or evaluation.
        Args:
            extra: Dict with additional information that can be provided by the FL system.
        """
        self.logger.info(f"Terminating {self.client_name} during {self.phase} phase.")
        if isinstance(self.trainer, monai.engines.Trainer):
            self.logger.info(f"Terminating {self.client_name} trainer...")
            self.trainer.terminate()
        if isinstance(self.evaluator, monai.engines.Trainer):
            self.logger.info(f"Terminating {self.client_name} evaluator...")
            self.evaluator.terminate()

        if self.multi_gpu:
            dist.destroy_process_group()

    def _check_converted(self, global_weights, local_var_dict, n_converted):
        if n_converted == 0:
            self.logger.warning(
                f"No global weights converted! Received weight dict keys are {list(global_weights.keys())}"
            )
        else:
            self.logger.info(
                f"Converted {n_converted} global variables to match {len(local_var_dict)} local variables."
            )

    def _set_cuda_device(self):
        if self.multi_gpu:
            self.rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.rank)

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
from typing import TYPE_CHECKING, Optional

import torch

import monai
from monai.bundle import ConfigParser
from monai.bundle.config_item import ConfigItem, ConfigComponent
from monai.config import IgniteInfo
from monai.fl.client.client_algo import ClientAlgo
from monai.fl.utils.constants import (
    BundleKeys,
    ExtraItems,
    FiltersType,
    FlPhase,
    FlStatistics,
    RequiredBundleKeys,
    WeightType,
    ModelType
)
from monai.fl.utils.exchange_object import ExchangeObject
from monai.networks.utils import copy_model_state, get_state_dict
from monai.utils import min_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Events
else:
    Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")


def convert_global_weights(global_weights, local_var_dict):
    """Helper function to convert global weights to local weights format"""
    # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
    model_keys = global_weights.keys()
    for var_name in local_var_dict:
        if var_name in model_keys:
            weights = global_weights[var_name]
            try:
                # reshape global weights to compute difference later on
                weights = torch.reshape(torch.as_tensor(weights), local_var_dict[var_name].shape)
                # update the local dict
                local_var_dict[var_name] = weights
            except Exception as e:
                raise ValueError(f"Convert weight from {var_name} failed.") from e
    return local_var_dict


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
        weight_diff[name] = local_var_dict[name].cpu() - global_weights[name]
        if torch.any(torch.isnan(weight_diff[name])):
            raise ValueError(f"Weights for {name} became NaN...")
    return weight_diff


def check_bundle_config(parser):
    for k in RequiredBundleKeys:
        if parser.get(k, None) is None:
            raise KeyError(f"Bundle config misses required key `{k}`")


def remove_ckpt_loader(parser):
    if BundleKeys.VALIDATE_HANDLERS in parser:
        for h in parser[BundleKeys.VALIDATE_HANDLERS]:
            if ConfigComponent.is_instantiable(h):
                if "CheckpointLoader" in h["_target_"]:
                    h["_disabled_"] = True


class MonaiAlgo(ClientAlgo):
    """
    Implementation of ``ClientAlgo`` to allow federated learning with MONAI bundle configurations.

    Args:
        bundle_root: path of bundle.
        local_epochs: number of local epochs to execute during each round of local training; defaults to 1.
        send_weight_diff: whether to send weight differences rather than full weights; defaults to `True`.
        config_train_filename: bundle training config path relative to bundle_root; defaults to "configs/train.json".
        config_evaluate_filename: bundle evaluation config path relative to bundle_root; defaults to "configs/evaluate.json".
        config_filters_filename: filter configuration file.
        disable_ckpt_loader: do not use CheckpointLoader if defined in train/evaluate configs; defaults to True.
        model_filepaths: dict of locations of best & final model checkpoints;
            defaults "model/model.pt" and "models/model_final.pt" relative to `bundle_root` for the best and final model, respectively.
    """

    def __init__(
        self,
        bundle_root: str,
        local_epochs: int = 1,
        send_weight_diff: bool = True,
        config_train_filename: Optional[str] = "configs/train.json",
        config_evaluate_filename: Optional[str] = "configs/evaluate.json",
        config_filters_filename: Optional[str] = None,
        disable_ckpt_loader: bool = True,
        model_filepaths: Optional[dict] = {ModelType.BEST_MODEL: "models/model.pt", ModelType.FINAL_MODEL: "models/model_final.pt"}
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.bundle_root = bundle_root
        self.local_epochs = local_epochs
        self.send_weight_diff = send_weight_diff
        self.config_train_filename = config_train_filename
        self.config_evaluate_filename = config_evaluate_filename
        self.config_filters_filename = config_filters_filename
        self.disable_ckpt_loader = disable_ckpt_loader
        self.model_filepaths = model_filepaths

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

        self.phase = FlPhase.IDLE
        self.client_name = None

    def initialize(self, extra=None):
        """
        Initialize routine to parse configuration files and extract main components such as trainer, evaluator, and filters.

        Args:
            extra: Dict with additional information that should be provided by FL system,
                i.e., `ExtraItems.CLIENT_NAME` and `ExtraItems.APP_ROOT`.

        """
        if extra is None:
            extra = {}
        self.logger.info(f"Initializing {self.client_name} ...")
        self.client_name = extra.get(ExtraItems.CLIENT_NAME, "noname")

        # FL platform needs to provide filepath to configuration files
        self.app_root = extra.get(ExtraItems.APP_ROOT, "")

        # Read bundle config files
        self.bundle_root = os.path.join(self.app_root, self.bundle_root)

        config_train_files = []
        config_eval_files = []
        config_filter_files = []
        if self.config_train_filename:
            config_train_files.append(os.path.join(self.bundle_root, self.config_train_filename))
            config_eval_files.append(os.path.join(self.bundle_root, self.config_train_filename))
        if self.config_evaluate_filename:
            config_eval_files.append(os.path.join(self.bundle_root, self.config_evaluate_filename))
        if self.config_filters_filename:  # filters are read by train_parser
            config_filter_files.append(os.path.join(self.bundle_root, self.config_filters_filename))

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
        if self.disable_ckpt_loader:
            remove_ckpt_loader(self.train_parser)
            remove_ckpt_loader(self.eval_parser)

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

        self.logger.info(f"Initialized {self.client_name}.")

    def train(self, data: ExchangeObject, extra=None):
        """
        Train on client's local data.

        Args:
            data: `ExchangeObject` containing the current global model weights.
            extra: Dict with additional information that can be provided by FL system.

        """
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
        self.global_weights = convert_global_weights(
            global_weights=data.weights, local_var_dict=get_state_dict(self.trainer.network)
        )

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
            extra: Dict with additional information that can be provided by FL system.

        Returns:
            return_weights: `ExchangeObject` containing current weights (default)
                or load requested model type from disk (`ModelType.BEST_MODEL` or `ModelType.FINAL_MODEL`).

        """
        if extra is None:
            extra = {}

        # by default return current weights, return best if requested via model type.
        self.phase = FlPhase.GET_WEIGHTS
        
        if ExtraItems.MODEL_TYPE in extra:
            model_type = extra.get(ExtraItems.MODEL_TYPE)
            if not isinstance(model_type, ModelType):
                raise ValueError(f"Expected requested model type to be of type `ModelType` but received {type(model_type)}")
            if model_type in self.model_filepaths:
                model_path = os.path.join(self.bundle_root, self.model_filepaths[model_type])
                if not os.path.isfile(model_path):
                    raise ValueError(f"No best model checkpoint exists at {model_path}")
                weights = torch.load(model_path, map_location="cpu")
                weigh_type = WeightType.WEIGHTS
                stats = dict()
                self.logger.info(f"Returning best checkpoint weights from {model_path}.")
            else:
                raise ValueError(f"Requested model type {model_type} not specified in `model_filepahts`: {self.model_filepaths}")
        else:
            if self.trainer:
                weights = get_state_dict(self.trainer.network)
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
            extra: Dict with additional information that can be provided by FL system.

        Returns:
            return_metrics: `ExchangeObject` containing evaluation metrics.

        """
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
        global_weights = convert_global_weights(
            global_weights=data.weights, local_var_dict=get_state_dict(self.evaluator.network)
        )

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
            extra: Dict with additional information that can be provided by FL system.
        """

        self.logger.info(f"Aborting {self.client_name} during {self.phase} phase.")
        if isinstance(self.trainer, monai.engines.Trainer):
            self.logger.info(f"Aborting {self.client_name} trainer...")
            self.trainer.terminate()
            self.trainer.state.dataloader_iter = self.trainer._dataloader_iter  # type: ignore
            if self.trainer.state.iteration % self.trainer.state.epoch_length == 0:
                self.trainer._fire_event(Events.EPOCH_COMPLETED)
        if isinstance(self.evaluator, monai.engines.Trainer):
            self.logger.info(f"Aborting {self.client_name} evaluator...")
            self.evaluator.terminate()

    def finalize(self, extra=None):
        """
        Finalize the training or evaluation.
        Args:
            extra: Dict with additional information that can be provided by FL system.
        """

        # TODO: finalize feature could be built into the MONAI Trainer class
        if extra is None:
            extra = {}
        self.logger.info(f"Terminating {self.client_name} during {self.phase} phase.")
        if isinstance(self.trainer, monai.engines.Trainer):
            self.logger.info(f"Terminating {self.client_name} trainer...")
            self.trainer.terminate()
        if isinstance(self.evaluator, monai.engines.Trainer):
            self.logger.info(f"Terminating {self.client_name} evaluator...")
            self.evaluator.terminate()

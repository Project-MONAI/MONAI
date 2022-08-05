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
import os.path
import sys
from typing import TYPE_CHECKING, Optional

import torch

import monai
from monai.bundle import ConfigParser
from monai.bundle.config_item import ConfigItem
from monai.config import IgniteInfo
from monai.engines.trainer import Trainer
from monai.fl.client.client_algo import ClientAlgo
from monai.fl.utils.constants import ExtraItems, FlPhase, WeightType, BundleConst, FiltersType
from monai.fl.utils.exchange_object import ExchangeObject
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
            try:
                # update the local dict
                local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
            except Exception as e:
                raise ValueError(f"Convert weight from {var_name} failed.") from e
    return local_var_dict


class MonaiAlgo(ClientAlgo):
    def __init__(
        self,
        config_train_file: Optional[str] = None,
        config_evaluate_file: Optional[str] = None,
        config_filters_file: Optional[str] = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_train_file = config_train_file
        self.config_evaluate_file = config_evaluate_file
        self.config_filters_file = config_filters_file

        self.parser = None
        self.trainer = None
        self.evaluator = None
        self.pre_filters = None
        self.post_weight_filters = None
        self.post_evaluate_filters = None

        self.phase = FlPhase.IDLE
        self.client_name = None

    def initialize(self, extra=None):
        if extra is None:
            extra = {}
        self.logger.info(f"Initializing {self.client_name} ...")
        self.client_name = extra.get(ExtraItems.CLIENT_NAME, "noname")

        # Read bundle config files
        config_files = []
        if self.config_train_file:
            config_files.append(self.config_train_file)
        if self.config_evaluate_file:
            config_files.append(self.config_evaluate_file)
        if self.config_filters_file:
            config_files.append(self.config_filters_file)

        # Parse
        self.parser = ConfigParser()
        self.parser.read_config(config_files)
        self.parser.parse()

        # Get trainer, evaluator, and filters
        self.trainer = self.parser.get_parsed_content(BundleConst.KEY_TRAINER, default=ConfigItem(None, BundleConst.KEY_TRAINER))
        self.evaluator = self.parser.get_parsed_content(BundleConst.KEY_EVALUATOR, default=ConfigItem(None, BundleConst.KEY_EVALUATOR))

        self.pre_filters = self.parser.get_parsed_content(FiltersType.PRE_FILTERS, default=ConfigItem(None, FiltersType.PRE_FILTERS))
        self.post_weight_filters = self.parser.get_parsed_content(FiltersType.POST_WEIGHT_FILTERS, default=ConfigItem(None, FiltersType.POST_WEIGHT_FILTERS))
        self.post_evaluate_filters = self.parser.get_parsed_content(FiltersType.POST_EVALUATE_FILTERS, default=ConfigItem(None, FiltersType.POST_EVALUATE_FILTERS))

        self.logger.info(f"Initialized {self.client_name}.")

    def train(self, data: ExchangeObject, extra=None):
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
        local_weights = convert_global_weights(
            global_weights=data.weights, local_var_dict=self.trainer.network.state_dict()
        )

        self.trainer.network.load_state_dict(local_weights)
        self.logger.info(f"Start {self.client_name} training...")
        self.trainer.run()

    def get_weights(self, extra=None):
        if extra is None:
            extra = {}
        self.phase = FlPhase.GET_WEIGHTS
        if self.trainer:
            weights = self.trainer.network.state_dict()
            stats = self.trainer.get_train_stats()  # TODO: returns dict with hardcoded strings from MONAI
        else:
            weights = None
            stats = dict()

        # TODO: support weight diffs
        if not isinstance(stats, dict):
            raise ValueError(f"stats is not a dict, {stats}")
        return_weights = ExchangeObject(
            weights=weights,
            optim=None,  # could be self.optimizer.state_dict()
            weight_type=WeightType.WEIGHTS,
            statistics=stats,
        )

        # filter weights if needed (use to apply differential privacy, encryption, compression, etc.)
        if self.post_weight_filters is not None:
            for _filter in self.post_weight_filters:
                return_weights = _filter(return_weights, extra)

        return return_weights

    def evaluate(self, data: ExchangeObject, extra=None):
        if extra is None:
            extra = {}
        if not isinstance(data, ExchangeObject):
            raise ValueError(f"expected data to be ExchangeObject but received {type(data)}")

        if self.predictor is None:
            raise ValueError("self.predictor should not be None.")
        if self.pre_filters is not None:
            for _filter in self.pre_filters:
                data = _filter(data, extra)

        self.phase = FlPhase.EVALUATE
        self.logger.info(f"Load {self.client_name} weights...")
        local_weights = convert_global_weights(
            global_weights=data.weights, local_var_dict=self.evaluator.network.state_dict()
        )

        self.predictor.network.load_state_dict(local_weights)
        self.logger.info(f"Start {self.client_name} evaluating...")
        self.predictor.run()
        return_metrics = ExchangeObject(metrics=self.predictor.state.metrics)

        if self.post_evaluate_filters is not None:
            for _filter in self.post_evaluate_filters:
                return_metrics = _filter(return_metrics, extra)
        return return_metrics

    def abort(self, extra=None):
        if extra is None:
            extra = {}
        self.logger.info(f"Aborting {self.client_name} during {self.phase} phase.")
        if isinstance(self.trainer, monai.engines.Trainer):
            self.logger.info(f"Aborting {self.client_name} trainer...")
            self.trainer.terminate()
            self.trainer.state.dataloader_iter = self.trainer._dataloader_iter  # type: ignore
            if self.trainer.state.iteration % self.trainer.state.epoch_length == 0:
                self.trainer._fire_event(Events.EPOCH_COMPLETED)
        if isinstance(self.predictor, monai.engines.Trainer):
            self.logger.info(f"Aborting {self.client_name} predictor...")
            self.predictor.terminate()

    def finalize(self, extra=None):
        # TODO: finalize feature could be built into the MONAI Trainer class

        if extra is None:
            extra = {}
        self.logger.info(f"Terminating {self.client_name} during {self.phase} phase.")
        if isinstance(self.trainer, monai.engines.Trainer):
            self.logger.info(f"Terminating {self.client_name} trainer...")
            self.trainer.terminate()
        if isinstance(self.predictor, monai.engines.Trainer):
            self.logger.info(f"Terminating {self.client_name} predictor...")
            self.predictor.terminate()

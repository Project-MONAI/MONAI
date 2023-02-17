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
from abc import ABC, abstractmethod
from logging.config import fileConfig
from pathlib import Path
from typing import Any, Sequence

import torch

from monai.apps.utils import get_logger
from monai.bundle.config_parser import ConfigParser
from monai.bundle.utils import DEFAULT_EXP_MGMT_SETTINGS
from monai.engines import Evaluator, Trainer
from monai.inferers import Inferer
from monai.transforms import Transform

__all__ = ["BundleWorkflow", "ConfigWorkflow", "ConfigTrainWorkflow", "ConfigInferWorkflow"]

logger = get_logger(module_name=__name__)


class BundleWorkflow(ABC):
    """
    Base class for the workflow specification in bundle.

    """

    @abstractmethod
    def initialize(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def finalize(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")


class TrainProperties:
    """
    Interface to get / set required properties for the training process in bundle.
    Subclass must implement the logic for properties.

    """

    @property
    def bundle_root(self) -> str:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @bundle_root.setter
    def bundle_root(self, path: str):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def device(self) -> torch.device:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @device.setter
    def device(self, name: str):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def dataset_dir(self) -> str:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @dataset_dir.setter
    def dataset_dir(self, path: str):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def trainer(self) -> Trainer:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @trainer.setter
    def trainer(self, trainer: Trainer | dict):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def max_epochs(self) -> int:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @max_epochs.setter
    def max_epochs(self, max_epochs: int):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def train_dataset(self) -> Any:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @train_dataset.setter
    def train_dataset(self, dataset: Any):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def train_dataset_data(self) -> Any:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @train_dataset_data.setter
    def train_dataset_data(self, data: Any):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def train_handlers(self) -> list:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @train_handlers.setter
    def train_handlers(self, handlers: list):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def train_inferer(self) -> Inferer:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @train_inferer.setter
    def train_inferer(self, inferer: Inferer | dict):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def train_preprocessing(self) -> Transform | None:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @train_preprocessing.setter
    def train_preprocessing(self, preprocessing: Transform | dict):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def train_postprocessing(self) -> Transform | None:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @train_postprocessing.setter
    def train_postprocessing(self, postprocessing: Transform | dict):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def train_key_metric(self) -> Any:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @train_key_metric.setter
    def train_key_metric(self, key_metric: Any):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def evaluator(self) -> Evaluator | None:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @evaluator.setter
    def evaluator(self, evaluator: Evaluator | dict):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def val_handlers(self) -> list | None:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @val_handlers.setter
    def val_handlers(self, handlers: list):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def val_dataset(self) -> Any:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @val_dataset.setter
    def val_dataset(self, dataset: Any):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def val_dataset_data(self) -> Any:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @val_dataset_data.setter
    def val_dataset_data(self, data: Any):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def val_inferer(self) -> Inferer | None:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @val_inferer.setter
    def val_inferer(self, inferer: Inferer | dict):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def val_preprocessing(self) -> Transform | None:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @val_preprocessing.setter
    def val_preprocessing(self, preprocessing: Transform | dict):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def val_postprocessing(self) -> Transform | None:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @val_postprocessing.setter
    def val_postprocessing(self, postprocessing: Transform | dict):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def val_key_metric(self) -> Any:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @val_key_metric.setter
    def val_key_metric(self, key_metric: Any):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")


class InferProperties:
    """
    Interface to get / set required properties for the inference process in bundle.
    Subclass must implement the logic for properties.

    """

    @property
    def bundle_root(self) -> str:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @bundle_root.setter
    def bundle_root(self, str):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def device(self) -> torch.device:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @device.setter
    def device(self, name: str):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def network_def(self) -> torch.Module:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @network_def.setter
    def network_def(self, net: dict):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def inferer(self) -> Inferer:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @inferer.setter
    def inferer(self, inferer: Inferer | dict):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def preprocessing(self) -> Transform | None:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @preprocessing.setter
    def preprocessing(self, preprocessing: Transform | dict):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def postprocessing(self) -> Transform | None:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @postprocessing.setter
    def postprocessing(self, postprocessing: Transform | dict):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @property
    def key_metric(self) -> Any:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @key_metric.setter
    def key_metric(self, key_metric: Any):
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")


class ConfigWorkflow(BundleWorkflow):
    def __init__(
        self,
        meta_file: str | Sequence[str] | None = None,
        config_file: str | Sequence[str] | None = None,
        logging_file: str | None = None,
        init_id: str = "initialize",
        run_id: str = "run",
        final_id: str = "finalize",
        tracking: str | dict | None = None,
        **override,
    ) -> None:
        if logging_file is not None:
            if not os.path.exists(logging_file):
                raise FileNotFoundError(f"can't find the logging config file: {logging_file}.")
            logger.info(f"set logging properties based on config: {logging_file}.")
            fileConfig(logging_file, disable_existing_loggers=False)

        self.parser = ConfigParser()
        self.parser.read_config(f=config_file)
        if meta_file is not None:
            self.parser.read_meta(f=meta_file)

        # the rest key-values in the _args are to override config content
        self.parser.update(pairs=override)
        self.init_id = init_id
        self.run_id = run_id
        self.final_id = final_id
        # set tracking configs for experiment management
        if tracking is not None:
            if isinstance(tracking, str) and tracking in DEFAULT_EXP_MGMT_SETTINGS:
                settings_ = DEFAULT_EXP_MGMT_SETTINGS[tracking]
            else:
                settings_ = ConfigParser.load_config_files(tracking)
            self.patch_bundle_tracking(parser=self.parser, settings=settings_)

    def initialize(self) -> bool:
        # reset the "reference_resolver" buffer at initialization stage
        self.parser.parse(reset=True)
        return self._run_expr(id=self.init_id)

    def run(self) -> bool:
        return self._run_expr(id=self.run_id)

    def finalize(self) -> bool:
        return self._run_expr(id=self.final_id)

    def _run_expr(self, id: str, **kwargs) -> bool:
        return self.parser.get_parsed_content(id, **kwargs) if id in self.parser else None

    def get_content(self, id: str, allow_missing: bool = False):
        if not self.parser.ref_resolver.is_resolved():
            raise RuntimeError("please execute 'initialize' before getting any parsed content.")
        if id not in self.parser:
            if allow_missing:
                return None
            else:
                raise KeyError(f"id '{id}' not in the config.")
        return self.parser.get_parsed_content(id=id)

    def set_content(self, id: str, content: Any, allow_missing: bool = False):
        if id not in self.parser:
            if allow_missing:
                return
            else:
                raise KeyError(f"id '{id}' not in the config.")
        self.parser[id] = content
        # must parse the config again after changing the content
        self.parser.ref_resolver.reset()

    def check(self) -> bool:
        pass

    def _check_required_ids(self, ids: Sequence[str]) -> bool:
        ret = True
        for i in ids:
            if i not in self.parser:
                logger.info(f"did not find the required id '{i}' in the config.")
                ret = False
        return ret

    def _check_optional_id(self, caller_id: str, expected: str):
        ret = self.parser.get(caller_id, None)
        if ret is not None and ret != "@" + expected:
            logger.info(f"found optional component with id '{ret}', but its id should be defined as `{expected}`.")
            return False
        return True

    @staticmethod
    def patch_bundle_tracking(parser: ConfigParser, settings: dict):
        """
        Patch the loaded bundle config with a new handler logic to enable experiment tracking features.

        Args:
            parser: loaded config content to patch the handler.
            settings: settings for the experiment tracking, should follow the pattern of default settings.

        """
        for k, v in settings["configs"].items():
            if k in settings["handlers_id"]:
                engine = parser.get(settings["handlers_id"][k]["id"])
                if engine is not None:
                    handlers = parser.get(settings["handlers_id"][k]["handlers"])
                    if handlers is None:
                        engine["train_handlers" if k == "trainer" else "val_handlers"] = [v]
                    else:
                        handlers.append(v)
            elif k not in parser:
                parser[k] = v
        # save the executed config into file
        default_name = f"config_{time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = parser.get("execute_config", None)
        if filepath is None:
            if "output_dir" not in parser:
                # if no "output_dir" in the bundle config, default to "<bundle root>/eval"
                parser["output_dir"] = "$@bundle_root + '/eval'"
            # experiment management tools can refer to this config item to track the config info
            parser["execute_config"] = parser["output_dir"] + f" + '/{default_name}'"
            filepath = os.path.join(parser.get_parsed_content("output_dir"), default_name)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        parser.export_config_file(parser.get(), filepath)


class ConfigTrainWorkflow(ConfigWorkflow, TrainProperties):
    required_train_ids = [
        "bundle_root",
        "device",
        "dataset_dir",
        "train#trainer",
        "train#trainer#max_epochs",
        "train#dataset",
        "train#dataset#data",
        "train#handlers",
        "train#inferer",
    ]
    required_val_ids = [
        "validate#evaluator",
        "validate#handlers",
        "validate#dataset",
        "validate#dataset#data",
        "validate#inferer",
    ]
    optional_train_ids = ["train#preprocessing", "train#postprocessing", "train#key_metric"]
    optional_val_ids = ["validate#preprocessing", "validate#postprocessing", "validate#key_metric"]

    def check(self) -> bool:
        ret = self._check_required_ids(self.required_train_ids)
        # if having validation logic, verify the ids
        ret &= self._check_required_ids(self.required_val_ids) if "validate" in self.parser else True
        # check optional ids if existing
        ret &= (
            self._check_optional_id("train#dataset#transform", self.optional_train_ids[0])
            & self._check_optional_id("train#trainer#postprocessing", self.optional_train_ids[1])
            & self._check_optional_id("train#trainer#key_train_metric", self.optional_train_ids[2])
            & self._check_optional_id("validate#dataset#transform", self.optional_val_ids[0])
            & self._check_optional_id("validate#evaluator#postprocessing", self.optional_val_ids[1])
            & self._check_optional_id("validate#evaluator#key_val_metric", self.optional_val_ids[2])
        )
        for h in self.parser["train#handlers"]:
            if h["_target_"] == "ValidationHandler":
                interval = h.get("interval", None)
                if interval is not None and interval != "val_interval":
                    logger.info(f"please use id 'val_interval' to define validation interval, got: '{interval}'.")
                    ret = False
        return ret

    @property
    def bundle_root(self) -> str:
        return self.get_content(self.required_train_ids[0])

    @bundle_root.setter
    def bundle_root(self, path: str):
        self.set_content(self.required_train_ids[0], path)

    @property
    def device(self) -> torch.device:
        return self.get_content(self.required_train_ids[1])

    @device.setter
    def device(self, name: str):
        self.set_content(self.required_train_ids[1], name)

    @property
    def dataset_dir(self) -> str:
        return self.get_content(self.required_train_ids[2])

    @dataset_dir.setter
    def dataset_dir(self, path: str):
        self.set_content(self.required_train_ids[2], path)

    @property
    def trainer(self) -> Trainer:
        return self.get_content(self.required_train_ids[3])

    @trainer.setter
    def trainer(self, trainer: Trainer | dict):
        self.set_content(self.required_train_ids[3], trainer)

    @property
    def max_epochs(self) -> int:
        return self.get_content(self.required_train_ids[4])

    @max_epochs.setter
    def max_epochs(self, max_epochs: int):
        self.set_content(self.required_train_ids[4], max_epochs)

    @property
    def train_dataset(self) -> Any:
        return self.get_content(self.required_train_ids[5])

    @train_dataset.setter
    def train_dataset(self, dataset: Any):
        self.set_content(self.required_train_ids[5], dataset)

    @property
    def train_dataset_data(self) -> Any:
        return self.get_content(self.required_train_ids[6])

    @train_dataset_data.setter
    def train_dataset_data(self, data: Any):
        self.set_content(self.required_train_ids[6], data)

    @property
    def train_handlers(self) -> list:
        return self.get_content(self.required_train_ids[7])

    @train_handlers.setter
    def train_handlers(self, handlers: list):
        self.set_content(self.required_train_ids[7], handlers)

    @property
    def train_inferer(self) -> Inferer:
        return self.get_content(self.required_train_ids[8])

    @train_inferer.setter
    def train_inferer(self, inferer: Inferer | dict):
        self.set_content(self.required_train_ids[8], inferer)

    @property
    def train_preprocessing(self) -> Transform | None:
        return self.get_content(self.optional_train_ids[0], allow_missing=True)

    @train_preprocessing.setter
    def train_preprocessing(self, preprocessing: Transform | dict):
        self.set_content(self.optional_train_ids[0], preprocessing, allow_missing=True)

    @property
    def train_postprocessing(self) -> Transform | None:
        return self.get_content(self.optional_train_ids[1], allow_missing=True)

    @train_postprocessing.setter
    def train_postprocessing(self, postprocessing: Transform | dict):
        self.set_content(self.optional_train_ids[1], postprocessing, allow_missing=True)

    @property
    def train_key_metric(self) -> Any:
        return self.get_content(self.optional_train_ids[2], allow_missing=True)

    @train_key_metric.setter
    def train_key_metric(self, key_metric: Any):
        self.set_content(self.optional_train_ids[2], key_metric, allow_missing=True)

    @property
    def evaluator(self) -> Evaluator | None:
        return self.get_content(self.required_val_ids[0], allow_missing=True)

    @evaluator.setter
    def evaluator(self, evaluator: Evaluator | dict):
        self.set_content(self.required_val_ids[0], evaluator, allow_missing=True)

    @property
    def val_handlers(self) -> list | None:
        return self.get_content(self.required_val_ids[1], allow_missing=True)

    @val_handlers.setter
    def val_handlers(self, handlers: list):
        self.set_content(self.required_val_ids[1], handlers, allow_missing=True)

    @property
    def val_dataset(self) -> Any:
        return self.get_content(self.required_val_ids[2], allow_missing=True)

    @val_dataset.setter
    def val_dataset(self, dataset: Any):
        self.set_content(self.required_val_ids[2], dataset, allow_missing=True)

    @property
    def val_dataset_data(self) -> Any:
        return self.get_content(self.required_val_ids[3], allow_missing=True)

    @val_dataset_data.setter
    def val_dataset_data(self, data: Any):
        self.set_content(self.required_val_ids[3], data, allow_missing=True)

    @property
    def val_inferer(self) -> Inferer | None:
        return self.get_content(self.required_val_ids[4], allow_missing=True)

    @val_inferer.setter
    def val_inferer(self, inferer: Inferer | dict):
        self.set_content(self.required_val_ids[4], inferer, allow_missing=True)

    @property
    def val_preprocessing(self) -> Transform | None:
        return self.get_content(self.optional_val_ids[0], allow_missing=True)

    @val_preprocessing.setter
    def val_preprocessing(self, preprocessing: Transform | dict):
        self.set_content(self.optional_val_ids[0], preprocessing, allow_missing=True)

    @property
    def val_postprocessing(self) -> Transform | None:
        return self.get_content(self.optional_val_ids[1], allow_missing=True)

    @val_postprocessing.setter
    def val_postprocessing(self, postprocessing: Transform | dict):
        self.set_content(self.optional_val_ids[1], postprocessing, allow_missing=True)

    @property
    def val_key_metric(self) -> Any:
        return self.get_content(self.optional_val_ids[2], allow_missing=True)

    @val_key_metric.setter
    def val_key_metric(self, key_metric: Any):
        self.set_content(self.optional_val_ids[2], key_metric, allow_missing=True)


class ConfigInferWorkflow(ConfigWorkflow, InferProperties):
    required_infer_ids = ["bundle_root", "device", "network_def", "inferer"]
    optional_infer_ids = ["preprocessing", "postprocessing", "key_metric"]

    def check(self) -> bool:
        ret = self._check_required_ids(self.infer_ids)
        # check optional ids if existing
        ret &= (
            self._check_optional_id("dataset#transform", self.optional_infer_ids[0])
            & self._check_optional_id("evaluator#postprocessing", self.optional_infer_ids[1])
            & self._check_optional_id("evaluator#key_val_metric", self.optional_infer_ids[2])
        )
        return ret

    @property
    def bundle_root(self) -> str:
        return self.get_content(self.required_infer_ids[0])

    @bundle_root.setter
    def bundle_root(self, path: str):
        self.set_content(self.required_infer_ids[0], path)

    @property
    def device(self) -> torch.device:
        return self.get_content(self.required_infer_ids[1])

    @device.setter
    def device(self, name: str):
        self.set_content(self.required_infer_ids[1], name)

    @property
    def network_def(self) -> torch.Module:
        return self.get_content(self.required_infer_ids[2])

    @network_def.setter
    def network_def(self, net: dict):
        self.set_content(self.required_infer_ids[2], net)

    @property
    def inferer(self) -> Inferer:
        return self.get_content(self.required_infer_ids[3])

    @inferer.setter
    def inferer(self, inferer: Inferer | dict):
        self.set_content(self.required_infer_ids[3], inferer)

    @property
    def preprocessing(self) -> Transform | None:
        return self.get_content(self.optional_infer_ids[0], allow_missing=True)

    @preprocessing.setter
    def preprocessing(self, preprocessing: Transform | dict):
        self.set_content(self.optional_infer_ids[0], preprocessing, allow_missing=True)

    @property
    def postprocessing(self) -> Transform | None:
        return self.get_content(self.optional_infer_ids[1], allow_missing=True)

    @postprocessing.setter
    def postprocessing(self, postprocessing: Transform | dict):
        self.set_content(self.optional_infer_ids[1], postprocessing, allow_missing=True)

    @property
    def key_metric(self) -> Any:
        return self.get_content(self.optional_infer_ids[2], allow_missing=True)

    @key_metric.setter
    def key_metric(self, key_metric: Any):
        self.set_content(self.optional_infer_ids[1], key_metric, allow_missing=True)

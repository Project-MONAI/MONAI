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

import os
import time
from abc import ABC, abstractmethod
from logging.config import fileConfig
from pathlib import Path
from typing import Any, Sequence
from monai.apps.utils import get_logger
from monai.bundle.config_parser import ConfigParser
from monai.bundle.utils import DEFAULT_EXP_MGMT_SETTINGS
from monai.engines import Trainer, Evaluator
from monai.inferers import Inferer

__all__ = ["BundleWorkflow", "ZooWorkflow", "ZooTrainWorkflow", "ZooInferWorkflow"]

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


class TrainProperties(ABC):
    """
    Interface to get / set required properties for the training process in bundle.
    Subclass must implement the logic for properties: "bundle_root", "device", "dataset_dir", "trainer",
    "max_epochs", "train_dataset", "train_dataset_data", "train_handlers", "val_evaluator", "val_handlers",
    "val_dataset", "val_dataset_data".

    """
    @abstractmethod
    @property
    def bundle_root(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @bundle_root.setter
    def bundle_root(self, path: str) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def device(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @device.setter
    def device(self, name: str) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def dataset_dir(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @dataset_dir.setter
    def dataset_dir(self, path: str) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def trainer(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @trainer.setter
    def trainer(self, trainer: Trainer | dict) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def max_epochs(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @max_epochs.setter
    def max_epochs(self, max_epochs: int) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def train_dataset(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @train_dataset.setter
    def train_dataset(self, dataset: Any) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def train_dataset_data(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @train_dataset_data.setter
    def train_dataset_data(self, data: Any) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def train_handlers(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @train_handlers.setter
    def train_handlers(self, handlers: list) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def evaluator(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @evaluator.setter
    def evaluator(self, evaluator: Evaluator | dict) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def val_handlers(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @val_handlers.setter
    def val_handlers(self, handlers: list) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def val_dataset(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @val_dataset.setter
    def val_dataset(self, dataset: Any) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def val_dataset_data(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @val_dataset_data.setter
    def val_dataset_data(self, data: Any) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")


class InferProperties(ABC):
    """
    Interface to get / set required properties for the inference process in bundle.
    Subclass must implement the logic for properties: "bundle_root", "device", "network_def", "inferer".

    """
    @abstractmethod
    @property
    def bundle_root(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @bundle_root.setter
    def bundle_root(self, str) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def device(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @device.setter
    def device(self, name: str) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def network_def(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @network_def.setter
    def network_def(self, net: dict) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @property
    def inferer(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @inferer.setter
    def inferer(self, inferer: Inferer | dict) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")


class ZooWorkflow(BundleWorkflow):
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

    def check(self) -> bool:
        pass

    @staticmethod
    def check_required_ids(ids: Sequence[str], parser: ConfigParser) -> bool:
        for i in ids:
            if i not in parser:
                logger.info(f"did not find the required id '{i}' in the config.")
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


class ZooTrainWorkflow(ZooWorkflow, TrainProperties):
    train_ids = [
        "bundle_root",
        "device",
        "dataset_dir",
        "train#trainer",
        "train#trainer#max_epochs",
        "train#dataset",
        "train#dataset#data",
        "train#handlers",
    ]
    val_ids = [
        "validate#evaluator",
        "validate#handlers",
        "validate#dataset",
        "validate#dataset#data",
    ]

    def check(self) -> bool:
        return self.check_required_ids(self.train_ids, self.parser) & \
            self.check_required_ids(self.val_ids, self.parser)

    @property
    def bundle_root(self) -> bool:
        return self.parser[self.train_ids[0]]

    @bundle_root.setter
    def bundle_root(self, path: str) -> bool:
        self.parser[self.train_ids[0]] = path

    @property
    def device(self) -> bool:
        return self.parser[self.train_ids[1]]

    @device.setter
    def device(self, name: str) -> bool:
        self.parser[self.train_ids[1]] = name

    @property
    def dataset_dir(self) -> bool:
        return self.parser[self.train_ids[2]]

    @dataset_dir.setter
    def dataset_dir(self, path: str) -> bool:
        self.parser[self.train_ids[2]] = path

    @property
    def trainer(self) -> bool:
        return self.parser[self.train_ids[3]]

    @trainer.setter
    def trainer(self, trainer: Trainer | dict) -> bool:
        self.parser[self.train_ids[3]] = trainer

    @property
    def max_epochs(self) -> bool:
        return self.parser[self.train_ids[4]]

    @max_epochs.setter
    def max_epochs(self, max_epochs: int) -> bool:
        self.parser[self.train_ids[4]] = max_epochs

    @property
    def train_dataset(self) -> bool:
        return self.parser[self.train_ids[5]]

    @train_dataset.setter
    def train_dataset(self, dataset: Any) -> bool:
        self.parser[self.train_ids[5]] = dataset

    @property
    def train_dataset_data(self) -> bool:
        return self.parser[self.train_ids[6]]

    @train_dataset_data.setter
    def train_dataset_data(self, data: Any) -> bool:
        self.parser[self.train_ids[6]] = data

    @property
    def train_handlers(self) -> bool:
        return self.parser[self.train_ids[7]]

    @train_handlers.setter
    def train_handlers(self, handlers: list) -> bool:
        self.parser[self.train_ids[7]] = handlers

    @property
    def evaluator(self) -> bool:
        return self.parser[self.val_ids[0]]

    @evaluator.setter
    def evaluator(self, evaluator: Evaluator | dict) -> bool:
        self.parser[self.val_ids[0]] = evaluator

    @property
    def val_handlers(self) -> bool:
        return self.parser[self.val_ids[1]]

    @val_handlers.setter
    def val_handlers(self, handlers: list) -> bool:
        self.parser[self.val_ids[1]] = handlers

    @property
    def val_dataset(self) -> bool:
        return self.parser[self.val_ids[2]]

    @val_dataset.setter
    def val_dataset(self, dataset: Any) -> bool:
        self.parser[self.val_ids[2]] = dataset

    @property
    def val_dataset_data(self) -> bool:
        return self.parser[self.val_ids[3]]

    @val_dataset_data.setter
    def val_dataset_data(self, data: Any) -> bool:
        self.parser[self.val_ids[3]] = data


class ZooInferWorkflow(ZooWorkflow, InferProperties):
    infer_ids = ["bundle_root", "device", "network_def", "inferer"]

    def check(self) -> bool:
        return self.check_required_keys(self.infer_ids, self.parser)

    @property
    def bundle_root(self) -> bool:
        return self.parser[self.infer_ids[0]]

    @bundle_root.setter
    def bundle_root(self, path: str) -> bool:
        self.parser[self.infer_ids[0]] = path

    @property
    def device(self) -> bool:
        return self.parser[self.infer_ids[1]]

    @device.setter
    def device(self, name: str) -> bool:
        self.parser[self.infer_ids[1]] = name

    @property
    def network_def(self) -> bool:
        return self.parser[self.infer_ids[2]]

    @network_def.setter
    def network_def(self, net: dict) -> bool:
        self.parser[self.infer_ids[2]] = net

    @property
    def inferer(self) -> bool:
        return self.parser[self.infer_ids[3]]

    @inferer.setter
    def inferer(self, inferer: Inferer | dict) -> bool:
        self.parser[self.infer_ids[3]] = inferer

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


class TrainAttributes(ABC):
    """
    Interface to get / set required atrributes for the training process in bundle.

    """
    @abstractmethod
    @property
    def bundle_root(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @bundle_root.setter
    def bundle_root(self, str) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")
    # FIXME: need to define all the other required attributes


class InferAttributes(ABC):
    """
    Interface to get / set required atrributes for the inference process in bundle.

    """
    @abstractmethod
    @property
    def bundle_root(self) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @bundle_root.setter
    def bundle_root(self, str) -> bool:
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")
    # FIXME: need to define all the other required attributes


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


class ZooTrainWorkflow(ZooWorkflow, TrainAttributes):
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
        return self.parser.bundle_root

    @bundle_root.setter
    def bundle_root(self, path: str) -> bool:
        self.parser["bundle_root"] = path


class ZooInferWorkflow(ZooWorkflow, InferAttributes):
    infer_ids = ["bundle_root", "device", "network_def", "inferer"]

    def check(self) -> bool:
        return self.check_required_keys(self.infer_ids, self.parser)

    @property
    def bundle_root(self) -> bool:
        return self.parser.bundle_root

    @bundle_root.setter
    def bundle_root(self, path: str) -> bool:
        self.parser["bundle_root"] = path

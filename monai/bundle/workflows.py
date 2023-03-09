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
from abc import ABC, abstractmethod
from logging.config import fileConfig
from pathlib import Path
from typing import Any, Sequence

from monai.apps.utils import get_logger
from monai.bundle.config_parser import ConfigParser
from monai.bundle.utils import DEFAULT_EXP_MGMT_SETTINGS
from monai.utils import BundleProperty, BundlePropertyConfig

__all__ = ["BundleWorkflow", "ConfigWorkflow"]

logger = get_logger(module_name=__name__)


TrainProperties = {
    "bundle_root": {
        BundleProperty.DESP: "root path of the bundle.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "bundle_root",
    },
    "device": {
        BundleProperty.DESP: "target device to execute the bundle workflow.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "device",
    },
    "train_preprocessing": {
        BundleProperty.DESP: "preprocessing for input data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "train#preprocessing",
        BundlePropertyConfig.REF_ID: "train#dataset#transform",
    },
    # TODO: add all the required and optional properties
}


InferProperties = {
    "bundle_root": {
        BundleProperty.DESP: "root path of the bundle.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "bundle_root",
    },
    "network_def": {
        BundleProperty.DESP: "network module for the inference.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "network_def",
    },
    "key_metric": {
        BundleProperty.DESP: "the key metric during evaluation.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "key_metric",
        BundlePropertyConfig.REF_ID: "evaluator#key_val_metric",
    },
    # TODO: add all the required and optional properties
}


class BundleWorkflow(ABC):
    """
    Base class for the workflow specification in bundle.

    Args:
        workflow: specifies the workflow type: "train" or "training" for a training workflow,
            or "infer", "inference", "eval", "evaluation" for a inference workflow,
            anything else will raise a ValueError.

    """

    def __init__(self, workflow: str):
        if workflow.lower() in ("train", "training"):
            self.properties = TrainProperties
        elif workflow.lower() in ("infer", "inference", "eval", "evaluation"):
            self.properties = InferProperties
        else:
            raise ValueError(f"Unsupported workflow type: '{workflow}'.")

    @abstractmethod
    def initialize(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def finalize(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def _get_property(self, name: str, property: dict) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def _set_property(self, name: str, property: dict, value: Any) -> Any:
        raise NotImplementedError()

    def __getattr__(self, name):
        if name in self.properties:
            return self._get_property(name=name, property=self.properties[name])
        else:
            return self.__getattribute__(name)  # getting regular attribute

    def __setattr__(self, name, value):
        if name != "properties" and name in self.properties:
            self._set_property(name=name, property=self.properties[name], value=value)
        else:
            super().__setattr__(name, value)  # setting regular attribute

    def check_properties(self):
        missing_props = [n for n, p in self.properties.items() if p[BundleProperty.REQUIRED] and not hasattr(self, n)]
        if missing_props:
            raise ValueError(f"Loaded bundle does not contain the following required properties: {missing_props}")


class ConfigWorkflow(BundleWorkflow):
    def __init__(
        self,
        workflow: str,
        config_file: str | Sequence[str],
        meta_file: str | Sequence[str] | None = "configs/metadata.json",
        logging_file: str | None = "configs/logging.conf",
        init_id: str = "initialize",
        run_id: str = "run",
        final_id: str = "finalize",
        tracking: str | dict | None = None,
        **override,
    ) -> None:
        super().__init__(workflow=workflow)
        if logging_file is not None:
            if not os.path.exists(logging_file):
                if logging_file == "configs/logging.conf":
                    warnings.warn("Default logging file in 'configs/logging.conf' does not exist, skipping logging.")
                else:
                    raise FileNotFoundError(f"Cannot find the logging config file: {logging_file}.")
            else:
                logger.info(f"Setting logging properties based on config: {logging_file}.")
                fileConfig(logging_file, disable_existing_loggers=False)

        self.parser = ConfigParser()
        self.parser.read_config(f=config_file)
        if meta_file is not None:
            if not os.path.exists(meta_file):
                if meta_file == "configs/metadata.json":
                    warnings.warn("Default metadata file in 'configs/metadata.json' does not exist, skipping loading.")
                else:
                    raise FileNotFoundError(f"Cannot find the metadata config file: {meta_file}.")
            else:
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

    def initialize(self) -> Any:
        # reset the "reference_resolver" buffer at initialization stage
        self.parser.parse(reset=True)
        return self._run_expr(id=self.init_id)

    def run(self) -> Any:
        return self._run_expr(id=self.run_id)

    def finalize(self) -> Any:
        return self._run_expr(id=self.final_id)

    def check_properties(self):
        super().check_properties()
        # also check whether the optional properties use correct ID name if existing
        wrong_props = []
        for n, p in self.properties.items():
            if not p[BundleProperty.REQUIRED] and not self._check_optional_id(name=n, property=p):
                wrong_props.append(n)
        if wrong_props:
            raise ValueError(f"Loaded bundle defines the following optional properties with wrong ID: {wrong_props}")
        # check validation `interval` property
        if "train#handlers" in self.parser:
            for h in self.parser["train#handlers"]:
                if h["_target_"] == "ValidationHandler":
                    interval = h.get("interval", None)
                    if interval not in (None, "val_interval"):
                        raise ValueError("Validation interval in training must be defined with ID 'val_interval'.")

    def _run_expr(self, id: str, **kwargs) -> Any:
        return self.parser.get_parsed_content(id, **kwargs) if id in self.parser else None

    def _get_id(self, name: str, property: dict) -> Any:
        pid = property[BundlePropertyConfig.ID]
        if pid not in self.parser:
            if not property[BundleProperty.REQUIRED]:
                return None
            else:
                raise KeyError(f"Property '{name}' with config ID '{pid}' not in the config.")
        return pid

    def _get_property(self, name: str, property: dict):
        if not self.parser.ref_resolver.is_resolved():
            raise RuntimeError("Please execute 'initialize' before getting any parsed content.")
        pid = self._get_id(name, property)
        return self.parser.get_parsed_content(id=pid) if pid is not None else None

    def _set_property(self, name: str, property: dict, value: Any):
        pid = self._get_id(name, property)
        if pid is not None:
            self.parser[pid] = value
            # must parse the config again after changing the content
            self.parser.ref_resolver.reset()

    def _check_optional_id(self, name: str, property: dict):
        if BundlePropertyConfig.REF_ID not in property:
            raise ValueError(f"Cannot find the ID of reference config item for optional property '{name}'.")
        ret = self.parser.get(property[BundlePropertyConfig.REF_ID], None)
        if ret is not None and ret != "@" + property[BundlePropertyConfig.ID]:
            return False
        return True

    # TODO: this function is called by MONAI FL, will update it after updated MONAI FL
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

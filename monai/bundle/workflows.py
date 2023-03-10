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
from monai.bundle.properties import InferProperties, TrainProperties
from monai.bundle.utils import DEFAULT_EXP_MGMT_SETTINGS
from monai.utils import BundleProperty, BundlePropertyConfig

__all__ = ["BundleWorkflow", "ConfigWorkflow"]

logger = get_logger(module_name=__name__)


class BundleWorkflow(ABC):
    """
    Base class for the workflow specification in bundle.

    Args:
        workflow: specifies the workflow type: "train" or "training" for a training workflow,
            or "infer", "inference", "eval", "evaluation" for a inference workflow,
            other unsupported string will raise a ValueError.
            default to `None` for common workflow.

    """

    def __init__(self, workflow: str | None = None):
        if workflow is None:
            self.properties = None
            self.workflow = None
            return
        if workflow.lower() in ("train", "training"):
            self.properties = TrainProperties
            self.workflow = "train"
        elif workflow.lower() in ("infer", "inference", "eval", "evaluation"):
            self.properties = InferProperties
            self.workflow = "infer"
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
        if self.properties is not None and name in self.properties:
            return self._get_property(name=name, property=self.properties[name])
        else:
            return self.__getattribute__(name)  # getting regular attribute

    def __setattr__(self, name, value):
        if name != "properties" and self.properties is not None and name in self.properties:
            self._set_property(name=name, property=self.properties[name], value=value)
        else:
            super().__setattr__(name, value)  # setting regular attribute

    def get_workflow_type(self):
        return self.workflow

    def check_properties(self) -> list[str] | None:
        if self.get_workflow_type() is None:
            warnings.warn("No available properties had been set, skipping check.")
            return None
        ret = [n for n, p in self.properties.items() if p.get(BundleProperty.REQUIRED, False) and not hasattr(self, n)]
        warnings.warn(f"Loaded bundle does not contain the following required properties: {ret}")
        return ret


class ConfigWorkflow(BundleWorkflow):
    def __init__(
        self,
        config_file: str | Sequence[str],
        meta_file: str | Sequence[str] | None = "configs/metadata.json",
        logging_file: str | None = "configs/logging.conf",
        init_id: str = "initialize",
        run_id: str = "run",
        final_id: str = "finalize",
        tracking: str | dict | None = None,
        workflow: str | None = None,
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

    def check_properties(self) -> list[str] | None:
        ret = super().check_properties()
        if ret is None:
            return None
        # also check whether the optional properties use correct ID name if existing
        wrong_props = []
        for n, p in self.properties.items():
            if not p.get(BundleProperty.REQUIRED, False) and not self._check_optional_id(name=n, property=p):
                wrong_props.append(n)
        if wrong_props:
            warnings.warn(f"Loaded bundle defines the following optional properties with wrong ID: {wrong_props}")
        ret.extend(wrong_props)
        return ret

    def _run_expr(self, id: str, **kwargs) -> Any:
        return self.parser.get_parsed_content(id, **kwargs) if id in self.parser else None

    def _get_prop_id(self, name: str, property: dict) -> Any:
        prop_id = property[BundlePropertyConfig.ID]
        if prop_id not in self.parser:
            if not property.get(BundleProperty.REQUIRED, False):
                return None
            else:
                raise KeyError(f"Property '{name}' with config ID '{prop_id}' not in the config.")
        return prop_id

    def _get_property(self, name: str, property: dict):
        if not self.parser.ref_resolver.is_resolved():
            raise RuntimeError("Please execute 'initialize' before getting any parsed content.")
        prop_id = self._get_prop_id(name, property)
        return self.parser.get_parsed_content(id=prop_id) if prop_id is not None else None

    def _set_property(self, name: str, property: dict, value: Any):
        prop_id = self._get_prop_id(name, property)
        if prop_id is not None:
            self.parser[prop_id] = value
            # must parse the config again after changing the content
            self.parser.ref_resolver.reset()

    def _check_optional_id(self, name: str, property: dict):
        id = property.get(BundlePropertyConfig.ID, None)
        ref_id = property.get(BundlePropertyConfig.REF_ID, None)
        if ref_id is None:
            # no ID of reference config item, skipping check for this optional property
            return True
        # check validation `validator` and `interval` properties as the handler index of ValidationHandler is unknown
        if name in ("evaluator", "val_interval"):
            if "train#handlers" in self.parser:
                for h in self.parser["train#handlers"]:
                    if h["_target_"] == "ValidationHandler":
                        arg_name = h.get(ref_id, None)
                        if arg_name != id:
                            return False
            return True
        ref = self.parser.get(ref_id, None)
        if ref is not None and ref != "@" + id:
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

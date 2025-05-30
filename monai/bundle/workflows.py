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

import json
import os
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from logging.config import fileConfig
from pathlib import Path
from typing import Any

from monai.apps.utils import get_logger
from monai.bundle.config_parser import ConfigParser
from monai.bundle.properties import InferProperties, MetaProperties, TrainProperties
from monai.bundle.utils import DEFAULT_EXP_MGMT_SETTINGS, EXPR_KEY, ID_REF_KEY, ID_SEP_KEY
from monai.config import PathLike
from monai.utils import BundleProperty, BundlePropertyConfig, ensure_tuple

__all__ = ["BundleWorkflow", "ConfigWorkflow"]

logger = get_logger(module_name=__name__)


class BundleWorkflow(ABC):
    """
    Base class for the workflow specification in bundle, it can be a training, evaluation or inference workflow.
    It defines the basic interfaces for the bundle workflow behavior: `initialize`, `run`, `finalize`, etc.
    And also provides the interface to get / set public properties to interact with a bundle workflow.

    Args:
        workflow_type: specifies the workflow type: "train" or "training" for a training workflow,
            or "infer", "inference", "eval", "evaluation" for a inference workflow,
            other unsupported string will raise a ValueError.
            default to `None` for only using meta properties.
        properties_path: the path to the JSON file of properties. If `workflow_type` is specified, properties will be
            loaded from the file based on the provided `workflow_type` and meta. If no `workflow_type` is specified,
            properties will default to loading from "meta". If `properties_path` is None, default properties
            will be sourced from "monai/bundle/properties.py" based on the workflow_type:
            For a training workflow, properties load from `TrainProperties` and `MetaProperties`.
            For a inference workflow, properties load from `InferProperties` and `MetaProperties`.
            For workflow_type = None : only `MetaProperties` will be loaded.
        meta_file: filepath of the metadata file, if this is a list of file paths, their contents will be merged in order.
        logging_file: config file for `logging` module in the program. for more details:
            https://docs.python.org/3/library/logging.config.html#logging.config.fileConfig.

    """

    supported_train_type: tuple = ("train", "training")
    supported_infer_type: tuple = ("infer", "inference", "eval", "evaluation")

    def __init__(
        self,
        workflow_type: str | None = None,
        properties_path: PathLike | None = None,
        meta_file: str | Sequence[str] | None = None,
        logging_file: str | None = None,
    ):
        if logging_file is not None:
            if not os.path.isfile(logging_file):
                raise FileNotFoundError(f"Cannot find the logging config file: {logging_file}.")
            logger.info(f"Setting logging properties based on config: {logging_file}.")
            fileConfig(logging_file, disable_existing_loggers=False)

        if meta_file is not None:
            if isinstance(meta_file, str) and not os.path.isfile(meta_file):
                logger.error(
                    f"Cannot find the metadata config file: {meta_file}. "
                    "Please see: https://docs.monai.io/en/stable/mb_specification.html"
                )
                meta_file = None
            if isinstance(meta_file, list):
                for f in meta_file:
                    if not os.path.isfile(f):
                        logger.error(
                            f"Cannot find the metadata config file: {f}. "
                            "Please see: https://docs.monai.io/en/stable/mb_specification.html"
                        )
                        meta_file = None

        if workflow_type is not None:
            if workflow_type.lower() in self.supported_train_type:
                workflow_type = "train"
            elif workflow_type.lower() in self.supported_infer_type:
                workflow_type = "infer"
            else:
                raise ValueError(f"Unsupported workflow type: '{workflow_type}'.")

        if properties_path is not None:
            properties_path = Path(properties_path)
            if not properties_path.is_file():
                raise ValueError(f"Property file {properties_path} does not exist.")
            with open(properties_path) as json_file:
                try:
                    properties = json.load(json_file)
                    self.properties: dict = {}
                    if workflow_type is not None and workflow_type in properties:
                        self.properties = properties[workflow_type]
                        if "meta" in properties:
                            self.properties.update(properties["meta"])
                    elif workflow_type is None:
                        if "meta" in properties:
                            self.properties = properties["meta"]
                            logger.info(
                                "No workflow type specified, default to load meta properties from property file."
                            )
                        else:
                            logger.warning("No 'meta' key found in properties while workflow_type is None.")
                except KeyError as e:
                    raise ValueError(f"{workflow_type} not found in property file {properties_path}") from e
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error decoding JSON from property file {properties_path}") from e
        else:
            if workflow_type == "train":
                self.properties = {**TrainProperties, **MetaProperties}
            elif workflow_type == "infer":
                self.properties = {**InferProperties, **MetaProperties}
            elif workflow_type is None:
                self.properties = copy(MetaProperties)
                logger.info("No workflow type and property file specified, default to 'meta' properties.")
            else:
                raise ValueError(f"Unsupported workflow type: '{workflow_type}'.")

        self.workflow_type = workflow_type
        self.meta_file = meta_file

    @abstractmethod
    def initialize(self, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize the bundle workflow before running.

        """
        raise NotImplementedError()

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the bundle workflow, it can be a training, evaluation or inference.

        """
        raise NotImplementedError()

    @abstractmethod
    def finalize(self, *args: Any, **kwargs: Any) -> Any:
        """
        Finalize step after the running of bundle workflow.

        """
        raise NotImplementedError()

    @abstractmethod
    def _get_property(self, name: str, property: dict) -> Any:
        """
        With specified property name and information, get the expected property value.

        Args:
            name: the name of target property.
            property: other information for the target property, defined in `TrainProperties` or `InferProperties`.

        """
        raise NotImplementedError()

    @abstractmethod
    def _set_property(self, name: str, property: dict, value: Any) -> Any:
        """
        With specified property name and information, set value for the expected property.

        Args:
            name: the name of target property.
            property: other information for the target property, defined in `TrainProperties` or `InferProperties`.
            value: value to set for the property.

        """
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
        """
        Get the workflow type, it can be `None`, "train", or "infer".

        """
        return self.workflow_type

    def get_meta_file(self):
        """
        Get the meta file.

        """
        return self.meta_file

    def add_property(self, name: str, required: str, desc: str | None = None) -> None:
        """
        Besides the default predefined properties, some 3rd party applications may need the bundle
        definition to provide additional properties for the specific use cases, if the bundle can't
        provide the property, means it can't work with the application.
        This utility adds the property for the application requirements check and access.

        Args:
            name: the name of target property.
            required: whether the property is "must-have".
            desc: descriptions for the property.
        """
        if self.properties is None:
            self.properties = {}
        if name in self.properties:
            logger.warning(f"property '{name}' already exists in the properties list, overriding it.")
        self.properties[name] = {BundleProperty.DESC: desc, BundleProperty.REQUIRED: required}

    def check_properties(self) -> list[str] | None:
        """
        Check whether the required properties are existing in the bundle workflow.
        If no workflow type specified, return None, otherwise, return a list of required but missing properties.

        """
        if self.properties is None:
            return None
        return [n for n, p in self.properties.items() if p.get(BundleProperty.REQUIRED, False) and not hasattr(self, n)]


class PythonicWorkflow(BundleWorkflow):
    """
    Base class for the pythonic workflow specification in bundle, it can be a training, evaluation or inference workflow.
    It defines the basic interfaces for the bundle workflow behavior: `initialize`, `finalize`, etc.
    This also provides the interface to get / set public properties to interact with a bundle workflow through
    defined `get_<property>` accessor methods or directly defining members of the object.
    For how to set the properties, users can define the `_set_<property>` methods or directly set the members of the object.
    The `initialize` method is called to set up the workflow before running. This method sets up internal state
    and prepares properties. If properties are modified after the workflow has been initialized, `self._is_initialized`
    is set to `False`. Before running the workflow again, `initialize` should be called to ensure that the workflow is
    properly set up with the new property values.

    Args:
        workflow_type: specifies the workflow type: "train" or "training" for a training workflow,
            or "infer", "inference", "eval", "evaluation" for a inference workflow,
            other unsupported string will raise a ValueError.
            default to `None` for only using meta properties.
        workflow: specifies the workflow type: "train" or "training" for a training workflow,
            or "infer", "inference", "eval", "evaluation" for a inference workflow,
            other unsupported string will raise a ValueError.
            default to `None` for common workflow.
        properties_path: the path to the JSON file of properties. If `workflow_type` is specified, properties will be
            loaded from the file based on the provided `workflow_type` and meta. If no `workflow_type` is specified,
            properties will default to loading from "meta". If `properties_path` is None, default properties
            will be sourced from "monai/bundle/properties.py" based on the workflow_type:
            For a training workflow, properties load from `TrainProperties` and `MetaProperties`.
            For a inference workflow, properties load from `InferProperties` and `MetaProperties`.
            For workflow_type = None : only `MetaProperties` will be loaded.
        config_file: path to the config file, typically used to store hyperparameters.
        meta_file: filepath of the metadata file, if this is a list of file paths, their contents will be merged in order.
        logging_file: config file for `logging` module in the program. for more details:
            https://docs.python.org/3/library/logging.config.html#logging.config.fileConfig.

    """

    supported_train_type: tuple = ("train", "training")
    supported_infer_type: tuple = ("infer", "inference", "eval", "evaluation")

    def __init__(
        self,
        workflow_type: str | None = None,
        properties_path: PathLike | None = None,
        config_file: str | Sequence[str] | None = None,
        meta_file: str | Sequence[str] | None = None,
        logging_file: str | None = None,
        **override: Any,
    ):
        meta_file = str(Path(os.getcwd()) / "metadata.json") if meta_file is None else meta_file
        super().__init__(
            workflow_type=workflow_type, properties_path=properties_path, meta_file=meta_file, logging_file=logging_file
        )
        self._props_vals: dict = {}
        self._set_props_vals: dict = {}
        self.parser = ConfigParser()
        if config_file is not None:
            self.parser.read_config(f=config_file)
        if self.meta_file is not None:
            self.parser.read_meta(f=self.meta_file)

        # the rest key-values in the _args are to override config content
        self.parser.update(pairs=override)
        self._is_initialized: bool = False

    def initialize(self, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize the bundle workflow before running.
        """
        self._props_vals = {}
        self._is_initialized = True

    def _get_property(self, name: str, property: dict) -> Any:
        """
        With specified property name and information, get the expected property value.
        If the property is already generated, return from the bucket directly.
        If user explicitly set the property, return it directly.
        Otherwise, generate the expected property as a class private property with prefix "_".

        Args:
            name: the name of target property.
            property: other information for the target property, defined in `TrainProperties` or `InferProperties`.
        """
        if not self._is_initialized:
            raise RuntimeError("Please execute 'initialize' before getting any properties.")
        value = None
        if name in self._set_props_vals:
            value = self._set_props_vals[name]
        elif name in self._props_vals:
            value = self._props_vals[name]
        elif name in self.parser.config[self.parser.meta_key]:  # type: ignore[index]
            id = self.properties.get(name, None).get(BundlePropertyConfig.ID, None)
            value = self.parser[id]
        else:
            try:
                value = getattr(self, f"get_{name}")()
            except AttributeError as e:
                if property[BundleProperty.REQUIRED]:
                    raise ValueError(
                        f"unsupported property '{name}' is required in the bundle properties,"
                        f"need to implement a method 'get_{name}' to provide the property."
                    ) from e
            self._props_vals[name] = value
        return value

    def _set_property(self, name: str, property: dict, value: Any) -> Any:
        """
        With specified property name and information, set value for the expected property.
        Stores user-reset initialized objects that should not be re-initialized and marks the workflow as not initialized.

        Args:
            name: the name of target property.
            property: other information for the target property, defined in `TrainProperties` or `InferProperties`.
            value: value to set for the property.

        """
        self._set_props_vals[name] = value
        self._is_initialized = False


class ConfigWorkflow(BundleWorkflow):
    """
    Specification for the config-based bundle workflow.
    Standardized the `initialize`, `run`, `finalize` behavior in a config-based training, evaluation, or inference.
    Before `run`, we add bundle root directory to Python search directories automatically.
    For more information: https://docs.monai.io/en/latest/mb_specification.html.

    Args:
        config_file: filepath of the config file, if this is a list of file paths, their contents will be merged in order.
        meta_file: filepath of the metadata file, if this is a list of file paths, their contents will be merged in order.
            If None, default to "configs/metadata.json", which is commonly used for bundles in MONAI model zoo.
        logging_file: config file for `logging` module in the program. for more details:
            https://docs.python.org/3/library/logging.config.html#logging.config.fileConfig.
            If None, default to "configs/logging.conf", which is commonly used for bundles in MONAI model zoo.
            If False, the logging logic for the bundle will not be modified.
        init_id: ID name of the expected config expression to initialize before running, default to "initialize".
            allow a config to have no `initialize` logic and the ID.
        run_id: ID name of the expected config expression to run, default to "run".
            to run the config, the target config must contain this ID.
        final_id: ID name of the expected config expression to finalize after running, default to "finalize".
            allow a config to have no `finalize` logic and the ID.
        tracking: if not None, enable the experiment tracking at runtime with optionally configurable and extensible.
            if "mlflow", will add `MLFlowHandler` to the parsed bundle with default tracking settings,
            if other string, treat it as file path to load the tracking settings.
            if `dict`, treat it as tracking settings.
            will patch the target config content with `tracking handlers` and the top-level items of `configs`.
            for detailed usage examples, please check the tutorial:
            https://github.com/Project-MONAI/tutorials/blob/main/experiment_management/bundle_integrate_mlflow.ipynb.
        workflow_type: specifies the workflow type: "train" or "training" for a training workflow,
            or "infer", "inference", "eval", "evaluation" for a inference workflow,
            other unsupported string will raise a ValueError.
            default to `None` for common workflow.
        properties_path: the path to the JSON file of properties. If `workflow_type` is specified, properties will be
            loaded from the file based on the provided `workflow_type` and meta. If no `workflow_type` is specified,
            properties will default to loading from "train". If `properties_path` is None, default properties
            will be sourced from "monai/bundle/properties.py" based on the workflow_type:
            For a training workflow, properties load from `TrainProperties` and `MetaProperties`.
            For a inference workflow, properties load from `InferProperties` and `MetaProperties`.
            For workflow_type = None : only `MetaProperties` will be loaded.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--net#input_chns 42``, ``--net %/data/other.json#net_arg``

    """

    def __init__(
        self,
        config_file: str | Sequence[str],
        meta_file: str | Sequence[str] | None = None,
        logging_file: str | bool | None = None,
        init_id: str = "initialize",
        run_id: str = "run",
        final_id: str = "finalize",
        tracking: str | dict | None = None,
        workflow_type: str | None = "train",
        properties_path: PathLike | None = None,
        **override: Any,
    ) -> None:
        if config_file is not None:
            _config_files = ensure_tuple(config_file)
            config_root_path = Path(_config_files[0]).parent
            for _config_file in _config_files:
                _config_file = Path(_config_file)
                if _config_file.parent != config_root_path:
                    logger.warning(
                        f"Not all config files are in {config_root_path}. If logging_file and meta_file are"
                        f"not specified, {config_root_path} will be used as the default config root directory."
                    )
                if not _config_file.is_file():
                    raise FileNotFoundError(f"Cannot find the config file: {_config_file}.")
        else:
            config_root_path = Path("configs")
        meta_file = str(config_root_path / "metadata.json") if meta_file is None else meta_file
        super().__init__(workflow_type=workflow_type, meta_file=meta_file, properties_path=properties_path)
        self.config_root_path = config_root_path
        logging_file = str(self.config_root_path / "logging.conf") if logging_file is None else logging_file
        if logging_file is False:
            logger.warning(f"Logging file is set to {logging_file}, skipping logging.")
        else:
            if not os.path.isfile(logging_file):
                if logging_file == str(self.config_root_path / "logging.conf"):
                    logger.warning(f"Default logging file in {logging_file} does not exist, skipping logging.")
                else:
                    raise FileNotFoundError(f"Cannot find the logging config file: {logging_file}.")
            else:
                fileConfig(str(logging_file), disable_existing_loggers=False)
                logger.info(f"Setting logging properties based on config: {logging_file}.")

        self.parser = ConfigParser()
        self.parser.read_config(f=config_file)
        if self.meta_file is not None:
            self.parser.read_meta(f=self.meta_file)
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
        self._is_initialized: bool = False

    def initialize(self) -> Any:
        """
        Initialize the bundle workflow before running.

        """
        # reset the "reference_resolver" buffer at initialization stage
        self.parser.parse(reset=True)
        self._is_initialized = True
        return self._run_expr(id=self.init_id)

    def run(self) -> Any:
        """
        Run the bundle workflow, it can be a training, evaluation or inference.
        Before run, we add bundle root directory to Python search directories automatically.

        """
        _bundle_root_path = (
            self.config_root_path.parent if self.config_root_path.name == "configs" else self.config_root_path
        )
        sys.path.insert(1, str(_bundle_root_path))
        if self.run_id not in self.parser:
            raise ValueError(f"run ID '{self.run_id}' doesn't exist in the config file.")
        return self._run_expr(id=self.run_id)

    def finalize(self) -> Any:
        """
        Finalize step after the running of bundle workflow.

        """
        return self._run_expr(id=self.final_id)

    def check_properties(self) -> list[str] | None:
        """
        Check whether the required properties are existing in the bundle workflow.
        If the optional properties have reference in the config, will also check whether the properties are existing.
        If no workflow type specified, return None, otherwise, return a list of required but missing properties.

        """
        ret = super().check_properties()
        if self.properties is None:
            logger.warning("No available properties had been set, skipping check.")
            return None
        if ret:
            logger.warning(f"Loaded bundle does not contain the following required properties: {ret}")
        # also check whether the optional properties use correct ID name if existing
        wrong_props = []
        for n, p in self.properties.items():
            if not p.get(BundleProperty.REQUIRED, False) and not self._check_optional_id(name=n, property=p):
                wrong_props.append(n)
        if wrong_props:
            logger.warning(f"Loaded bundle defines the following optional properties with wrong ID: {wrong_props}")
        if ret is not None:
            ret.extend(wrong_props)
        return ret

    def _run_expr(self, id: str, **kwargs: dict) -> list[Any]:
        """
        Evaluate the expression or expression list given by `id`. The resolved values from the evaluations are not stored,
        allowing this to be evaluated repeatedly (eg. in streaming applications) without restarting the hosting process.
        """
        ret = []
        if id in self.parser:
            # suppose all the expressions are in a list, run and reset the expressions
            if isinstance(self.parser[id], list):
                for i in range(len(self.parser[id])):
                    sub_id = f"{id}{ID_SEP_KEY}{i}"
                    ret.append(self.parser.get_parsed_content(sub_id, **kwargs))
                    self.parser.ref_resolver.remove_resolved_content(sub_id)
            else:
                ret.append(self.parser.get_parsed_content(id, **kwargs))
                self.parser.ref_resolver.remove_resolved_content(id)
        return ret

    def _get_prop_id(self, name: str, property: dict) -> Any:
        prop_id = property[BundlePropertyConfig.ID]
        if prop_id not in self.parser:
            if not property.get(BundleProperty.REQUIRED, False):
                return None
            else:
                raise KeyError(f"Property '{name}' with config ID '{prop_id}' not in the config.")
        return prop_id

    def _get_property(self, name: str, property: dict) -> Any:
        """
        With specified property name and information, get the parsed property value from config.

        Args:
            name: the name of target property.
            property: other information for the target property, defined in `TrainProperties` or `InferProperties`.

        """
        if not self._is_initialized:
            raise RuntimeError("Please execute 'initialize' before getting any parsed content.")
        prop_id = self._get_prop_id(name, property)
        return self.parser.get_parsed_content(id=prop_id) if prop_id is not None else None

    def _set_property(self, name: str, property: dict, value: Any) -> None:
        """
        With specified property name and information, set value for the expected property.

        Args:
            name: the name of target property.
            property: other information for the target property, defined in `TrainProperties` or `InferProperties`.
            value: value to set for the property.

        """
        prop_id = self._get_prop_id(name, property)
        if prop_id is not None:
            self.parser[prop_id] = value
            # must parse the config again after changing the content
            self._is_initialized = False
            self.parser.ref_resolver.reset()

    def add_property(  # type: ignore[override]
        self, name: str, required: str, config_id: str, desc: str | None = None
    ) -> None:
        """
        Besides the default predefined properties, some 3rd party applications may need the bundle
        definition to provide additional properties for the specific use cases, if the bundle can't
        provide the property, means it can't work with the application.
        This utility adds the property for the application requirements check and access.

        Args:
            name: the name of target property.
            required: whether the property is "must-have".
            config_id: the config ID of target property in the bundle definition.
            desc: descriptions for the property.

        """
        super().add_property(name=name, required=required, desc=desc)
        self.properties[name][BundlePropertyConfig.ID] = config_id

    def _check_optional_id(self, name: str, property: dict) -> bool:
        """
        If an optional property has reference in the config, check whether the property is existing.
        If `ValidationHandler` is defined for a training workflow, will check whether the optional properties
        "evaluator" and "val_interval" are existing.

        Args:
            name: the name of target property.
            property: other information for the target property, defined in `TrainProperties` or `InferProperties`.

        """
        id = property.get(BundlePropertyConfig.ID, None)
        ref_id = property.get(BundlePropertyConfig.REF_ID, None)
        if ref_id is None:
            # no ID of reference config item, skipping check for this optional property
            return True
        # check validation `validator` and `interval` properties as the handler index of ValidationHandler is unknown
        ref: str | None = None
        if name in ("evaluator", "val_interval"):
            if f"train{ID_SEP_KEY}handlers" in self.parser:
                for h in self.parser[f"train{ID_SEP_KEY}handlers"]:
                    if h["_target_"] == "ValidationHandler":
                        ref = h.get(ref_id, None)
        else:
            ref = self.parser.get(ref_id, None)
        # for reference IDs that not refer to a property directly but using expressions, skip the check
        if ref is not None and not ref.startswith(EXPR_KEY) and ref != ID_REF_KEY + id:
            return False
        return True

    @staticmethod
    def patch_bundle_tracking(parser: ConfigParser, settings: dict) -> None:
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
        # Users can set the `save_execute_config` to `False`, `/path/to/artifacts` or `True`.
        # If set to False, nothing will be recorded. If set to True, the default path will be logged.
        # If set to a file path, the given path will be logged.
        filepath = parser.get("save_execute_config", True)
        if filepath:
            if isinstance(filepath, bool):
                if "output_dir" not in parser:
                    # if no "output_dir" in the bundle config, default to "<bundle root>/eval"
                    parser["output_dir"] = f"{EXPR_KEY}{ID_REF_KEY}bundle_root + '/eval'"
                # experiment management tools can refer to this config item to track the config info
                parser["save_execute_config"] = parser["output_dir"] + f" + '/{default_name}'"
                filepath = os.path.join(parser.get_parsed_content("output_dir"), default_name)
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            parser.export_config_file(parser.get(), filepath)
        else:
            parser["save_execute_config"] = None

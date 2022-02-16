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

import inspect
import sys
import warnings
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Dict, List, Optional, Sequence, Union

from monai.apps.mmars.utils import find_refs_in_config, is_instantiable, resolve_config_with_refs
from monai.utils import ensure_tuple, instantiate

__all__ = ["ComponentLocator", "ConfigItem", "ConfigComponent"]


class ComponentLocator:
    """
    Scan all the available classes and functions in the MONAI package and map them with the module paths in a table.
    It's used to locate the module path for provided component name.

    Args:
        excludes: if any string of the `excludes` exists in the full module name, don't import this module.

    """

    MOD_START = "monai"

    def __init__(self, excludes: Optional[Union[Sequence[str], str]] = None):
        self.excludes = [] if excludes is None else ensure_tuple(excludes)
        self._components_table: Optional[Dict[str, List]] = None

    def _find_module_names(self) -> List[str]:
        """
        Find all the modules start with MOD_START and don't contain any of `excludes`.

        """
        return [
            m for m in sys.modules.keys() if m.startswith(self.MOD_START) and all(s not in m for s in self.excludes)
        ]

    def _find_classes_or_functions(self, modnames: Union[Sequence[str], str]) -> Dict[str, List]:
        """
        Find all the classes and functions in the modules with specified `modnames`.

        Args:
            modnames: names of the target modules to find all the classes and functions.

        """
        table: Dict[str, List] = {}
        # all the MONAI modules are already loaded by `load_submodules`
        for modname in ensure_tuple(modnames):
            try:
                # scan all the classes and functions in the module
                module = import_module(modname)
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) or inspect.isfunction(obj)) and obj.__module__ == modname:
                        if name not in table:
                            table[name] = []
                        table[name].append(modname)
            except ModuleNotFoundError:
                pass
        return table

    def get_component_module_name(self, name) -> Optional[Union[List[str], str]]:
        """
        Get the full module name of the class / function with specified name.
        If target component name exists in multiple packages or modules, return a list of full module names.

        Args:
            name: name of the expected class or function.

        """
        if self._components_table is None:
            # init component and module mapping table
            self._components_table = self._find_classes_or_functions(self._find_module_names())

        mods: Optional[Union[List[str], str]] = self._components_table.get(name, None)
        if isinstance(mods, list) and len(mods) == 1:
            mods = mods[0]
        return mods


class ConfigItem:
    """
    Utility class to manage every item of the whole config content.
    When recursively parsing a complicated config content, every item (like: dict, list, string, int, float, etc.)
    can be treated as an "config item" then construct a `ConfigItem`.
    For example, below are 5 config items when recursively parsing:
    - a dict: `{"preprocessing": ["@transform1", "@transform2", "$lambda x: x"]}`
    - a list: `["@transform1", "@transform2", "$lambda x: x"]`
    - a string: `"transform1"`
    - a string: `"transform2"`
    - a string: `"$lambda x: x"`

    `ConfigItem` can set optional unique ID name, then another config item may refer to it.
    The references mean the IDs of other config items used as "@XXX" in the current config item, for example:
    config item with ID="A" is a list `[1, 2, 3]`, another config item can be `"args": {"input_list": "@A"}`.
    If sub-item in the config is instantiable, also treat it as reference because must instantiate it before
    resolving current config.
    It can search the config content and find out all the references, and resolve the config content
    when all the references are resolved.

    Here we predefined 3 kinds special marks (`#`, `@`, `$`) when parsing the whole config content:
    - "XXX#YYY": join nested config IDs, like "transforms#5" is ID name of the 6th transform in a list ID="transforms".
    - "@XXX": current config item refers to another config item XXX, like `{"args": {"data": "@dataset"}}` uses
    resolved config content of `dataset` as the parameter "data".
    - "$XXX": execute the string after "$" as python code with `eval()` function, like "$@model.parameters()".

    The typical usage of the APIs:
    - Initialize with config content.
    - If having references, get the IDs of its referring components.
    - When all the referring components are resolved, resolve the config content with them,
    and execute expressions in the config.

    .. code-block:: python

        config = {"lr": "$@epoch / 1000"}

        configer = ConfigComponent(config, id="test")
        dep_ids = configer.get_id_of_refs()
        configer.resolve_config(refs={"epoch": 10})
        lr = configer.get_resolved_config()

    Args:
        config: content of a config item, can be a `dict`, `list`, `string`, `float`, `int`, etc.
        id: ID name of current config item, useful to construct referring config items.
            for example, config item A may have ID "transforms#A" and config item B refers to A
            and uses the resolved config content of A as an arg `{"args": {"other": "@transforms#A"}}`.
            `id` defaults to `None`, if some component refers to current component, `id` must be a `string`.
        globals: to support executable string in the config, sometimes we need to provide the global variables
            which are referred in the executable string. for example: `globals={"monai": monai} will be useful
            for config `{"collate_fn": "$monai.data.list_data_collate"}`.

    """

    def __init__(self, config: Any, id: Optional[str] = None, globals: Optional[Dict] = None) -> None:
        self.config = config
        self.resolved_config = None
        self.is_resolved = False
        self.id = id
        self.globals = globals
        self.update_config(config=config)

    def get_id(self) -> Optional[str]:
        """
        ID name of current config item, useful to construct referring config items.
        for example, config item A may have ID "transforms#A" and config item B refers to A
        and uses the resolved config content of A as an arg `{"args": {"other": "@transforms#A"}}`.
        `id` defaults to `None`, if some component refers to current component, `id` must be a `string`.

        """
        return self.id

    def update_config(self, config: Any):
        """
        Update the config content for a config item at runtime.
        If having references, need resolve the config later.
        A typical usage is to modify the initial config content at runtime and set back.

        Args:
            config: content of a config item, can be a `dict`, `list`, `string`, `float`, `int`, etc.

        """
        self.config = config
        self.resolved_config = None
        self.is_resolved = False
        if not self.get_id_of_refs():
            # if no references, can resolve the config immediately
            self.resolve(refs=None)

    def get_config(self):
        """
        Get the initial config content of current config item, usually set at the constructor.
        It can be useful to dynamically update the config content before resolving.

        """
        return self.config

    def get_id_of_refs(self) -> List[str]:
        """
        Recursively search all the content of current config item to get the IDs of references.
        It's used to detect and resolve all the references before resolving current config item.
        For `dict` and `list`, recursively check the sub-items.
        For example: `{"args": {"lr": "$@epoch / 1000"}}`, the reference IDs: `["epoch"]`.

        """
        return find_refs_in_config(self.config, id=self.id)

    def resolve(self, refs: Optional[Dict] = None):
        """
        If all the references are resolved in `refs`, resolve the config content with them to construct `resolved_config`.

        Args:
            refs: all the resolved referring items with ID as keys, default to `None`.

        """
        self.resolved_config = resolve_config_with_refs(self.config, id=self.id, refs=refs, globals=self.globals)
        self.is_resolved = True

    def get_resolved_config(self):
        """
        Get the resolved config content, constructed in `resolve_config()`. The returned config has no references,
        then use it in the program, for example: initial config item `{"intervals": "@epoch / 10"}` and references
        `{"epoch": 100}`, the resolved config will be `{"intervals": 10}`.

        """
        return self.resolved_config


class Instantiable(ABC):
    """
    Base class for instantiable object with module name and arguments.

    """

    @abstractmethod
    def resolve_module_name(self, *args: Any, **kwargs: Any):
        """
        Utility function to resolve the target module name.

        """
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def resolve_args(self, *args: Any, **kwargs: Any):
        """
        Utility function to resolve the arguments.

        """
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def is_disabled(self, *args: Any, **kwargs: Any):
        """
        Utility function to check whether the target component is disabled.

        """
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def instantiate(self, *args: Any, **kwargs: Any):
        """
        Instantiate the target component.

        """
        raise NotImplementedError(f"subclass {self.__class__.__name__} must implement this method.")


class ConfigComponent(ConfigItem, Instantiable):
    """
    Subclass of :py:class:`monai.apps.ConfigItem`, the config item represents a target component of class
    or function, and support to instantiate the component. Example of config item:
    `{"<name>": "LoadImage", "<args>": {"keys": "image"}}`

    Here we predefined 4 keys: `<name>`, `<path>`, `<args>`, `<disabled>` for component config:
    - '<name>' - class / function name in the modules of packages.
    - '<path>' - directly specify the module path, based on PYTHONPATH, ignore '<name>' if specified.
    - '<args>' - arguments to initialize the component instance.
    - '<disabled>' - if defined `'<disabled>': True`, will skip the buiding, useful for development or tuning.

    The typical usage of the APIs:
    - Initialize with config content.
    - If no references, `instantiate` the component if having "<name>" or "<path>" keywords and return the instance.
    - If having references, get the IDs of its referring components.
    - When all the referring components are resolved, resolve the config content with them, execute expressions in
    the config and `instantiate`.

    .. code-block:: python

        locator = ComponentLocator(excludes=["<not_needed_modules>"])
        config = {"<name>": "DataLoader", "<args>": {"dataset": "@dataset", "batch_size": 2}}

        configer = ConfigComponent(config, id="test_config", locator=locator)
        configer.resolve_config(refs={"dataset": Dataset(data=[1, 2])})
        configer.get_resolved_config()
        dataloader: DataLoader = configer.instantiate()

    Args:
        config: content of a component config item, should be a dict with `<name>` or `<path>` key.
        id: ID name of current config item, useful to construct referring config items.
            for example, config item A may have ID "transforms#A" and config item B refers to A
            and uses the resolved config content of A as an arg `{"args": {"other": "@transforms#A"}}`.
            `id` defaults to `None`, if some component refers to current component, `id` must be a `string`.
        locator: `ComponentLocator` to help locate the module path of `<name>` in the config and instantiate.
            if `None`, will create a new `ComponentLocator` with specified `excludes`.
        excludes: if `locator` is None, create a new `ComponentLocator` with `excludes`. any string of the `excludes`
            exists in the module name, don't import this module.
        globals: to support executable string in the config, sometimes we need to provide the global variables
            which are referred in the executable string. for example: `globals={"monai": monai} will be useful
            for config `"collate_fn": "$monai.data.list_data_collate"`.

    """

    def __init__(
        self,
        config: Any,
        id: Optional[str] = None,
        locator: Optional[ComponentLocator] = None,
        excludes: Optional[Union[Sequence[str], str]] = None,
        globals: Optional[Dict] = None,
    ) -> None:
        super().__init__(config=config, id=id, globals=globals)
        self.locator = ComponentLocator(excludes=excludes) if locator is None else locator

    def resolve_module_name(self):
        """
        Utility function used in `instantiate()` to resolve the target module name from provided config content.
        The config content must have `<path>` or `<name>`.

        """
        config = self.get_resolved_config()
        path = config.get("<path>", None)
        if path is not None:
            if "<name>" in config:
                warnings.warn(f"should not set both '<path>' and '<name>', default to use '<path>': {path}.")
            return path

        name = config.get("<name>", None)
        if name is None:
            raise ValueError("must provide `<path>` or `<name>` of target component to instantiate.")

        module = self.locator.get_component_module_name(name)
        if module is None:
            raise ModuleNotFoundError(f"can not find component '{name}' in {self.locator.MOD_START} modules.")
        if isinstance(module, list):
            warnings.warn(
                f"there are more than 1 component have name `{name}`: {module}, use the first one `{module[0]}."
                f" if want to use others, please set its module path in `<path>` directly."
            )
            module = module[0]
        return f"{module}.{name}"

    def resolve_args(self):
        """
        Utility function used in `instantiate()` to resolve the arguments from config content of target component.

        """
        return self.get_resolved_config().get("<args>", {})

    def is_disabled(self):
        """
        Utility function used in `instantiate()` to check whether the target component is disabled.

        """
        return self.get_resolved_config().get("<disabled>", False)

    def instantiate(self, **kwargs) -> object:  # type: ignore
        """
        Instantiate component based on the resolved config content.
        The target component must be a `class` or a `function`, otherwise, return `None`.

        Args:
            kwargs: args to override / add the config args when instantiation.

        """
        if not self.is_resolved:
            warnings.warn(
                "the config content of current component has not been resolved,"
                " please try to resolve the references first."
            )
            return None
        if not is_instantiable(self.get_resolved_config()) or self.is_disabled():
            # if not a class or function or marked as `disabled`, skip parsing and return `None`
            return None

        modname = self.resolve_module_name()
        args = self.resolve_args()
        args.update(kwargs)
        return instantiate(modname, **args)

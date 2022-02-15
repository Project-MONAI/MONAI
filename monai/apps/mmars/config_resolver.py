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
from importlib import import_module
from typing import Any, Dict, List, Optional, Sequence, Union

from monai.apps.mmars.utils import able_to_build, resolve_config_with_deps, search_config_with_deps
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

    `ConfigItem` can set optional unique ID name, then another config item may depdend on it, for example:
    config item with ID="A" is a list `[1, 2, 3]`, another config item can be `"args": {"input_list": "@A"}`.
    It can search the config content and find out all the dependencies, and resolve the config content
    when all the dependencies are resolved.

    Here we predefined 3 kinds special marks (`#`, `@`, `$`) when parsing the whole config content:
    - "XXX#YYY": join nested config IDs, like "transforms#5" is ID name of the 6th transform in a list ID="transforms".
    - "@XXX": current config item depends on another config item XXX, like `{"args": {"data": "@dataset"}}` uses
    resolved config content of `dataset` as the parameter "data".
    - "$XXX": execute the string after "$" as python code with `eval()` function, like "$@model.parameters()".

    The typical usage of the APIs:
    - Initialize with config content.
    - If having dependencies, get the IDs of its dependent components.
    - When all the dependent components are resolved, resolve the config content with them,
    and execute expressions in the config.

    .. code-block:: python

        config = {"lr": "$@epoch / 1000"}

        configer = ConfigComponent(config, id="test")
        dep_ids = configer.get_id_of_deps()
        configer.resolve_config(deps={"epoch": 10})
        lr = configer.get_resolved_config()

    Args:
        config: content of a config item, can be a `dict`, `list`, `string`, `float`, `int`, etc.
        id: ID name of current config item, useful to construct dependent config items.
            for example, config item A may have ID "transforms#A" and config item B depends on A
            and uses the resolved config content of A as an arg `{"args": {"other": "@transforms#A"}}`.
            `id` defaults to `None`, if some component depends on current component, `id` must be a `string`.
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
        self.set_config(config=config)

    def get_id(self) -> Optional[str]:
        """
        ID name of current config item, useful to construct dependent config items.
        for example, config item A may have ID "transforms#A" and config item B depends on A
        and uses the resolved config content of A as an arg `{"args": {"other": "@transforms#A"}}`.
        `id` defaults to `None`, if some component depends on current component, `id` must be a `string`.

        """
        return self.id

    def set_config(self, config: Any):
        """
        Set the config content for a config item at runtime.
        If having dependencies, need resolve the config later.
        A typical usage is to modify the initial config content at runtime and set back.

        Args:
            config: content of a config item, can be a `dict`, `list`, `string`, `float`, `int`, etc.

        """
        self.config = config
        self.resolved_config = None
        self.is_resolved = False
        if not self.get_id_of_deps():
            # if no dependencies, can resolve the config immediately
            self.resolve_config(deps=None)

    def get_config(self):
        """
        Get the initial config content of current config item, usually set at the constructor.
        It can be useful to dynamically update the config content before resolving.

        """
        return self.config

    def get_id_of_deps(self) -> List[str]:
        """
        Recursively search all the content of current config item to get the IDs of dependencies.
        It's used to detect and resolve all the dependencies before resolving current config item.
        For `dict` and `list`, recursively check the sub-items.
        For example: `{"args": {"lr": "$@epoch / 1000"}}`, the dependency IDs: `["epoch"]`.

        """
        return search_config_with_deps(config=self.config, id=self.id)

    def resolve_config(self, deps: Optional[Dict] = None):
        """
        If all the dependencies are resolved in `deps`, resolve the config content with them to construct `resolved_config`.

        Args:
            deps: all the resolved dependent items with ID as keys, default to `None`.

        """
        self.resolved_config = resolve_config_with_deps(config=self.config, deps=deps, id=self.id, globals=self.globals)
        self.is_resolved = True

    def get_resolved_config(self):
        """
        Get the resolved config content, constructed in `resolve_config()`. The returned config has no dependencies,
        then use it in the program, for example: initial config item `{"intervals": "@epoch / 10"}` and dependencies
        `{"epoch": 100}`, the resolved config will be `{"intervals": 10}`.

        """
        return self.resolved_config


class ConfigComponent(ConfigItem):
    """
    Subclass of :py:class:`monai.apps.ConfigItem`, the config item represents a target component of class
    or function, and support to build the instance. Example of config item:
    `{"<name>": "LoadImage", "<args>": {"keys": "image"}}`

    Here we predefined 4 keys: `<name>`, `<path>`, `<args>`, `<disabled>` for component config:
    - '<name>' - class / function name in the modules of packages.
    - '<path>' - directly specify the module path, based on PYTHONPATH, ignore '<name>' if specified.
    - '<args>' - arguments to initialize the component instance.
    - '<disabled>' - if defined `'<disabled>': True`, will skip the buiding, useful for development or tuning.

    The typical usage of the APIs:
    - Initialize with config content.
    - If no dependencies, `build` the component if having "<name>" or "<path>" keywords and return the instance.
    - If having dependencies, get the IDs of its dependent components.
    - When all the dependent components are resolved, resolve the config content with them, execute expressions in
    the config and `build` instance.

    .. code-block:: python

        locator = ComponentLocator(excludes=["<not_needed_modules>"])
        config = {"<name>": "DataLoader", "<args>": {"dataset": "@dataset", "batch_size": 2}}

        configer = ConfigComponent(config, id="test_config", locator=locator)
        configer.resolve_config(deps={"dataset": Dataset(data=[1, 2])})
        configer.get_resolved_config()
        dataloader: DataLoader = configer.build()

    Args:
        config: content of a component config item, should be a dict with `<name>` or `<path>` key.
        id: ID name of current config item, useful to construct dependent config items.
            for example, config item A may have ID "transforms#A" and config item B depends on A
            and uses the resolved config content of A as an arg `{"args": {"other": "@transforms#A"}}`.
            `id` defaults to `None`, if some component depends on current component, `id` must be a `string`.
        locator: `ComponentLocator` to help locate the module path of `<name>` in the config and build instance.
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

    def _resolve_module_name(self):
        """
        Utility function used in `build()` to resolve the target module name from provided config content.
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
            raise ValueError("must provide `<path>` or `<name>` of target component to build.")

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

    def _resolve_args(self):
        """
        Utility function used in `build()` to resolve the arguments from config content of target component to build.

        """
        return self.get_resolved_config().get("<args>", {})

    def _is_disabled(self):
        """
        Utility function used in `build()` to check whether the target component is disabled building.

        """
        return self.get_resolved_config().get("<disabled>", False)

    def build(self, **kwargs) -> object:
        """
        Build component instance based on the resolved config content.
        The target component must be a `class` or a `function`, otherwise, return `None`.

        Args:
            kwargs: args to override / add the config args when building.

        """
        if not self.is_resolved:
            warnings.warn(
                "the config content of current component has not been resolved,"
                " please try to resolve the dependencies first."
            )
            return None
        config = self.get_resolved_config()
        if not able_to_build(config) or self._is_disabled():
            # if not a class or function, or marked as `disabled`, skip parsing and return `None`
            return None

        modname = self._resolve_module_name()
        args = self._resolve_args()
        args.update(kwargs)
        return instantiate(modname, **args)

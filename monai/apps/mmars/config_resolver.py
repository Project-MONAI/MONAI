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

from asyncio import FastChildWatcher
import inspect
import sys
import warnings
from importlib import import_module
from typing import Any, Dict, List, Optional, Sequence, Union

from monai.apps.mmars.utils import is_to_build, resolve_config_with_deps, search_config_with_deps
from monai.utils import ensure_tuple, instantiate

__all__ = ["ComponentLocator", "ConfigComponent"]


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
        self._components_table = None

    def _find_module_names(self) -> List[str]:
        return [m for m in sys.modules.keys() if m.startswith(self.MOD_START) and all(s not in m for s in self.excludes)]

    def _find_classes_or_functions(self, modnames: Union[Sequence[str], str]):
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

    def get_component_module_name(self, name) -> Union[List[str], str]:
        """
        Get the full module name of the class / function with specified name.
        If target component name exists in multiple packages or modules, return a list of full module names.

        Args:
            name: name of the expected class or function.

        """
        if self._components_table is None:
            # init component and module mapping table
            self._components_table = self._find_classes_or_functions(self._find_module_names())

        mods = self._components_table.get(name, None)
        if isinstance(mods, list) and len(mods) == 1:
            mods = mods[0]
        return mods


class ConfigComponent:
    """
    Utility class to manage every component in the config with a unique `id` name.
    When recursively parsing a complicated config dictionary, every item should be treated as a `ConfigComponent`.
    For example:
    - `{"preprocessing": [{"<name>": "LoadImage", "<args>": {"keys": "image"}}]}`
    - `{"<name>": "LoadImage", "<args>": {"keys": "image"}}`
    - `"<name>": "LoadImage"`
    - `"keys": "image"`

    It can search the config content and find out all the dependencies, then build the config to instance
    when all the dependencies are resolved.

    Here we predefined 4 kinds special marks (`<>`, `#`, `@`, `$`) to parse the config content:
    - "<XXX>": like "<name>" is the name of a target component, to distinguish it with regular key "name"
    in the config content. now we have 4 keys: `<name>`, `<path>`, `<args>`, `<disabled>`.
    - "XXX#YYY": join nested config ids, like "transforms#5" is id name of the 6th transform in the transforms list.
    - "@XXX": use an component as config item, like `"input_data": "@dataset"` uses `dataset` instance as parameter.
    - "$XXX": execute the string after "$" as python code with `eval()` function, like "$@model.parameters()".

    The typical usage of the APIs:
    - Initialize with config content.
    - If no dependencies, `build` the component if having "<name>" or "<path>" keywords and return the instance.
    - If having dependencies, get the IDs of its dependent components.
    - When all the dependent components are built, update the config content with them, execute expressions in
    the config and `build` instance.

    Args:
        config: config content of current component, can be a `dict`, `list`, `string`, `float`, `int`, etc.
        no_deps: flag to mark whether the config has dependent components, default to `False`. if `True`,
            no need to resolve dependencies before building.
        id: ID name of current config component, useful to construct dependent components.
            for example, component A may have ID "transforms#A" and component B depends on A
            and uses the built instance of A as a dependent arg `"XXX": "@transforms#A"`.
            for nested config items, use `#` to join ids, for list component, use index from `0` as id.
            for example: `transform`, `transform#5`, `transform#5#<args>#keys`, etc.
            the ID can be useful to quickly get the expected item in a complicated and nested config content.
            ID defaults to `None`, if some component depends on current component, ID must be a `string`.
        locator: ComponentLocator to help locate the module path of `<name>` in the config and build instance.
            if `None`, will create a new ComponentLocator with specified `excludes`.
        excludes: if `locator` is None, create a new ComponentLocator with `excludes`. any string of the `excludes`
            exists in the full module name, don't import this module.
        globals: to support executable string in the config, sometimes we need to provide the global variables
            which are referred in the executable string. for example: `globals={"monai": monai} will be useful
            for config `"collate_fn": "$monai.data.list_data_collate"`.

    """

    def __init__(
        self,
        config: Any,
        no_deps: bool = False,
        id: Optional[str] = None,
        locator: Optional[ComponentLocator] = None,
        excludes: Optional[Union[Sequence[str], str]] = None,
        globals: Optional[Dict] = None,
    ) -> None:
        self.config = None
        self.resolved_config = None
        self.is_resolved = False
        self.id = id
        self.locator = ComponentLocator(excludes=excludes) if locator is None else locator
        self.globals = globals
        self.set_config(config=config, no_deps=no_deps)

    def get_id(self) -> str:
        """
        Get the unique ID of current component, useful to construct dependent components.
        For example, component A may have ID "transforms#A" and component B depends on A
        and uses the built instance of A as a dependent arg `"XXX": "@transforms#A"`.
        ID defaults to `None`, if some component depends on current component, ID must be a string.

        """
        return self.id

    def set_config(self, config: Any, no_deps: bool = False):
        self.config = config
        self.resolved_config = None
        self.is_resolved = False
        if no_deps:
            # if no dependencies, can resolve the config immediately
            self.resolve_config(deps=None)

    def get_config(self):
        """
        Get the init config content of current config component, usually set at the constructor.
        It can be useful for lazy instantiation to dynamically update the config content before resolving

        """
        return self.config

    def get_id_of_deps(self) -> List[str]:
        """
        Recursively search all the content of current config compoent to get the ids of dependencies.
        It's used to build all the dependencies before build current config component.
        For `dict` and `list`, treat every item as a dependency.
        For example, for `{"<name>": "DataLoader", "<args>": {"dataset": "@dataset"}}`, the dependency ids:
        `["<name>", "<args>", "<args>#dataset", "dataset"]`.

        """
        return search_config_with_deps(config=self.config, id=self.id)

    def resolve_config(self, deps: dict):
        """
        If all the dependencies are ready in `deps`, update the config content with them and return new config.
        It can be used for lazy instantiation, the returned config has no dependencies, can be built immediately.

        Args:
            deps: all the dependent components with ids.

        """
        self.resolved_config = resolve_config_with_deps(config=self.config, deps=deps, id=self.id, globals=self.globals)
        self.is_resolved = True

    def get_resolved_config(self):
        return self.resolved_config

    def _resolve_module_name(self):
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
                f"there are more than 1 component name `{name}`: {module}, use the first one `{module[0]}."
                f" if want to use others, please set the full python path in `<path>` directly."
            )
            module = module[0]
        return f"{module}.{name}"

    def _resolve_args(self):
        return self.get_resolved_config().get("<args>", {})

    def _is_disabled(self):
        return self.get_resolved_config().get("<disabled>", False)

    def build(self, **kwargs) -> object:
        """
        Build component instance based on the resolved config content.
        The target component must be a `class` or a `function`.
        Supported special keys for the config:
        - '<name>' - class / function name in the modules of packages.
        - '<path>' - directly specify the path, based on PYTHONPATH, ignore '<name>' if specified.
        - '<args>' - arguments to initialize the component instance.
        - '<disabled>' - if defined `'<disabled>': True`, will skip the buiding, useful for development or tuning.

        Args:
            kwargs: args to override / add the config args when building.

        Raises:
            ValueError: must provide `<path>` or `<name>` of class / function to build component.
            ValueError: can not find component class or function.

        """
        if not self.is_resolved:
            warnings.warn(
                "the config content of current component has not been resolved,"
                " please try to resolve the dependencies first."
            )
        config = self.get_resolved_config()
        if not is_to_build(config) or self._is_disabled():
            # if not a class or function, or marked as `disabled`, skip parsing and return `None`
            return None

        modname = self._resolve_module_name()
        args = self._resolve_args()
        args.update(kwargs)
        return instantiate(modname, **args)

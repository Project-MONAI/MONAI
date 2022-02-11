# Copyright 2020 - 2021 MONAI Consortium
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
import warnings
from importlib import import_module
from pkgutil import walk_packages
from typing import Any, Dict, List, Optional, Sequence

from monai.apps.mmars.utils import search_configs_with_deps, update_configs_with_deps
from monai.utils.module import instantiate

__all__ = ["ComponentScanner", "ConfigComponent"]


class ComponentScanner:
    """
    Scan all the available classes and functions in the specified packages and modules.
    Map the all the names and the module names in a table.

    Args:
        pkgs: the expected packages to scan modules and parse component names in the config.
        modules: the expected modules in the packages to scan for all the components.
            for example, to parser "LoadImage" in config, `pkgs` can be ["monai"], `modules` can be ["transforms"].
        excludes: if any string of the `excludes` exists in the full module name, don't import this module.

    """

    def __init__(
        self, pkgs: Sequence[str], modules: Optional[Sequence[str]] = None, excludes: Optional[Sequence[str]] = None
    ):
        for p in pkgs:
            if not p.startswith("monai"):
                raise ValueError("only support to scan MONAI package so far.")
        self.pkgs = pkgs
        self.modules = [] if modules is None else modules
        self.excludes = [] if excludes is None else excludes
        self._components_table = self._create_table()

    def _create_table(self):
        table: Dict[str, List] = {}
        for pkg in self.pkgs:
            package = import_module(pkg)

            for _, modname, _ in walk_packages(path=package.__path__, prefix=package.__name__ + "."):
                # if no modules specified, load all modules in the package
                if all(s not in modname for s in self.excludes) and (
                    len(self.modules) == 0 or any(name in modname for name in self.modules)
                ):
                    try:
                        module = import_module(modname)
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) or inspect.isfunction(obj)) and obj.__module__ == modname:
                                if name not in table:
                                    table[name] = []
                                table[name].append(modname)
                    except ModuleNotFoundError:
                        pass
        return table

    def get_component_module_name(self, name):
        """
        Get the full module name of the class / function with specified name.
        If target component name exists in multiple packages or modules, return all the paths.z

        Args:
            name: name of the expected class or function.

        """
        mods = self._components_table.get(name, None)
        if isinstance(mods, list) and len(mods) == 1:
            mods = mods[0]
        return mods


class ConfigComponent:
    """
    Utility class to manage every component in the config with a unique `id` name.
    When recursively parsing a complicated config dictioanry, every item should be treated as a `ConfigComponent`.
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

    Args:
        id: id name of current config component, for nested config items, use `#` to join ids.
            for list component, use index from `0` as id.
            for example: `transform`, `transform#5`, `transform#5#<args>#keys`, etc.
            the id can be useful to quickly get the expected item in a complicated and nested config content.
        config: config content of current component, can be a `dict`, `list`, `string`, `float`, `int`, etc.
        scanner: ComponentScanner to help get the `<name>` or `<path>` in the config and build instance.
        globals: to support executable string in the config, sometimes we need to provide the global variables
            which are referred in the executable string. for example: `globals={"monai": monai} will be useful
            for config `"collate_fn": "$monai.data.list_data_collate"`.

    """

    def __init__(self, id: str, config: Any, scanner: ComponentScanner, globals: Optional[Dict] = None) -> None:
        self.id = id
        self.config = config
        self.scanner = scanner
        self.globals = globals

    def get_id(self) -> str:
        """
        Get the id name of current config component.

        """
        return self.id

    def get_config(self):
        """
        Get the raw config content of current config component.

        """
        return self.config

    def get_dependent_ids(self) -> List[str]:
        """
        Recursively search all the content of current config compoent to get the ids of dependencies.
        It's used to build all the dependencies before build current config component.
        For `dict` and `list`, treat every item as a dependency.
        For example, for `{"<name>": "DataLoader", "<args>": {"dataset": "@dataset"}}`, the dependency ids:
        `["<name>", "<args>", "<args>#dataset", "dataset"]`.

        """
        return search_configs_with_deps(config=self.config, id=self.id)

    def get_updated_config(self, deps: dict):
        """
        If all the dependencies are ready in `deps`, update the config content with them and return new config.
        It can be used for lazy instantiation.

        Args:
            deps: all the dependent components with ids.

        """
        return update_configs_with_deps(config=self.config, deps=deps, id=self.id, globals=self.globals)

    def _check_dependency(self, config):
        """
        Check whether current config still has unresolved dependencies or executable string code.

        Args:
            config: config content to check.

        """
        if isinstance(config, list):
            for i in config:
                if self._check_dependency(i):
                    return True
        if isinstance(config, dict):
            for v in config.values():
                if self._check_dependency(v):
                    return True
        if isinstance(config, str):
            if config.startswith("&") or "@" in config:
                return True
        return False

    def build(self, config: Optional[Dict] = None, **kwargs) -> object:
        """
        Build component instance based on the provided dictonary config.
        The target component must be a class and a function.
        Supported special keys for the config:
        - '<name>' - class / function name in the modules of packages.
        - '<path>' - directly specify the path, based on PYTHONPATH, ignore '<name>' if specified.
        - '<args>' - arguments to initialize the component instance.
        - '<disabled>' - if defined `'<disabled>': True`, will skip the buiding, useful for development or tuning.

        Args:
            config: dictionary config that defines a component.
            kwargs: args to override / add the config args when building.

        Raises:
            ValueError: must provide `<path>` or `<name>` of class / function to build component.
            ValueError: can not find component class or function.

        """
        config = self.config if config is None else config
        if self._check_dependency(config=config):
            warnings.warn("config content has other dependencies or executable string, skip `build`.")
            return config

        if (
            not isinstance(config, dict)
            or ("<name>" not in config and "<path>" not in config)
            or config.get("<disabled>") is True
        ):
            # if marked as `disabled`, skip parsing
            return config

        args = config.get("<args>", {})
        args.update(kwargs)
        path = self._get_path(config)
        return instantiate(path, **args)

    def _get_path(self, config):
        """
        Get the path of class / function specified in the config content.

        Args:
            config: dictionary config that defines a component.

        """
        path = config.get("<path>", None)
        if path is None:
            name = config.get("<name>", None)
            if name is None:
                raise ValueError("must provide `<path>` or `<name>` of target component to build.")
            module = self.scanner.get_component_module_name(name)
            if module is None:
                raise ValueError(f"can not find component '{name}'.")
            if isinstance(module, list):
                warnings.warn(
                    f"there are more than 1 component name `{name}`: {module}, use the first one `{module[0]}."
                    f" if want to use others, please set the full python path in `<path>` directly."
                )
                module = module[0]
            path = f"{module}.{name}"

        return path

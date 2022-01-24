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

from typing import Any, Dict, List, Optional, Sequence
import importlib
import inspect
import pkgutil
from xmlrpc.client import FastMarshaller

from torch import warnings

from monai.apps.mmars.utils import instantiate_class, search_configs_with_objs, update_configs_with_objs


class ModuleScanner:
    """
    Scan all the available classes in the specified packages and modules.
    Map the all the class names and the module names in a table.

    Args:
        pkgs: the expected packages to scan.
        modules: the expected modules in the packages to scan.

    """

    def __init__(self, pkgs: Sequence[str], modules: Sequence[str]):
        self.pkgs = pkgs
        self.modules = modules
        self._class_table = self._create_classes_table()

    def _create_classes_table(self):
        class_table = {}
        for pkg in self.pkgs:
            package = importlib.import_module(pkg)

            for _, modname, _ in pkgutil.walk_packages(path=package.__path__, prefix=package.__name__ + "."):
                # if no modules specified, load all modules in the package
                if len(self.modules) == 0 or any(name in modname for name in self.modules):
                    try:
                        module = importlib.import_module(modname)
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and obj.__module__ == modname:
                                class_table[name] = modname
                    except ModuleNotFoundError:
                        pass
        return class_table

    def get_class_module_name(self, class_name):
        return self._class_table.get(class_name, None)


class ConfigComponent:
    def __init__(self, id: str, config: Any, module_scanner: ModuleScanner, globals: Optional[Dict] = None) -> None:
        self.id = id
        self.config = config
        self.module_scanner = module_scanner
        self.globals = globals

    def get_id(self) -> str:
        return self.id

    def get_config(self):
        return self.config

    def get_referenced_ids(self) -> List[str]:
        return search_configs_with_objs(self.config, [], id=self.id)

    def get_updated_config(self, refs: dict):
        return update_configs_with_objs(config=self.config, refs=refs, id=self.id, globals=self.globals)

    def _check_dependency(self, config):
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

    def build(self, config: Optional[Dict] = None) -> object:
        """
        Build component instance based on the provided dictonary config.
        Supported special keys for the config:
        - '<name>' - class name in the modules of packages.
        - '<path>' - directly specify the class path, based on PYTHONPATH, ignore '<name>' if specified.
        - '<args>' - arguments to initialize the component instance.
        - '<disabled>' - if defined `'<disabled>': true`, will skip the buiding, useful for development or tuning.

        Args:
            config: dictionary config to define a component.

        Raises:
            ValueError: must provide `path` or `name` of class to build component.
            ValueError: can not find component class.

        """
        config = self.config if config is None else config
        if self._check_dependency(config=config):
            warnings.warn("config content has other dependencies or executable string, skip `build`.")
            return config

        if not isinstance(config, dict) \
            or ("<name>" not in config and "<path>" not in config) \
            or config.get("<disabled>") is True:
            # if marked as `disabled`, skip parsing
            return config

        class_args = config.get("<args>", {})
        class_path = self._get_class_path(config)
        return instantiate_class(class_path, **class_args)

    def _get_class_path(self, config):
        class_path = config.get("<path>", None)
        if class_path is None:
            class_name = config.get("<name>", None)
            if class_name is None:
                raise ValueError("must provide `<path>` or `<name>` of class to build component.")
            module_name = self.module_scanner.get_class_module_name(class_name)
            if module_name is None:
                raise ValueError(f"can not find component class '{class_name}'.")
            class_path = f"{module_name}.{class_name}"

        return class_path


class ConfigResolver:
    def __init__(self, components: Optional[Dict[str, ConfigComponent]] = None):
        self.resolved_configs = {}
        self.resolved_components = {}
        self.components = {} if components is None else components

    def add(self, component: ConfigComponent):
        id = component.get_id()
        if id in self.components:
            raise ValueError(f"id '{id}' is already added.")
        self.components[id] = component

    def _resolve_one_component(self, id: str, instantiate: bool = True) -> bool:
        com = self.components[id]
        # check whether the obj has any unresolved refs in its args
        ref_ids = com.get_referenced_ids()
        refs = {}
        if len(ref_ids) > 0:
            # see whether all refs are resolved
            for comp_id in ref_ids:
                if comp_id not in self.resolved_components:
                    # this referenced component is not resolved
                    if comp_id not in self.components:
                        raise RuntimeError(f"the reference component `{comp_id}` is not in config.")
                    # resolve the dependency first
                    self._resolve_one_component(id=comp_id, instantiate=True)
                refs[comp_id] = self.resolved_components[comp_id]
            # all referenced components are resolved already
        updated_config = com.get_updated_config(refs)
        resolved_com = None

        if instantiate:
            resolved_com = com.build(updated_config)
            self.resolved_configs[id] = updated_config
            self.resolved_components[id] = resolved_com

        return updated_config, resolved_com

    def resolve_all(self):
        for k in self.components.keys():
            self._resolve_one_component(id=k, instantiate=True)

    def get_resolved_compnent(self, id: str):
        if id not in self.resolved_components:
            self._resolve_one_component(id=id, instantiate=True)
        return self.resolved_components[id]

    def get_resolved_config(self, id: str):
        if id not in self.resolved_configs:
            config, _ = self._resolve_one_component(id=id, instantiate=False)
        else:
            config = self.resolved_configs[id]
        return config

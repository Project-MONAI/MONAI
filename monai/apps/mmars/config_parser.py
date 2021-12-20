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

import importlib
import inspect
import pkgutil
from typing import Dict, Sequence

from monai.apps.mmars.utils import instantiate_class


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
                if any(name in modname for name in self.modules):
                    try:
                        module = importlib.import_module(modname)
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and obj.__module__ == modname:
                                class_table[name] = modname
                    except ModuleNotFoundError:
                        pass
        return class_table

    def get_module_name(self, class_name):
        return self._class_table.get(class_name, None)


class ConfigParser:
    """
    Parse dictionary format config and build components.

    Args:
        pkgs: the expected packages to scan.
        modules: the expected modules in the packages to scan.

    """

    def __init__(self, pkgs: Sequence[str], modules: Sequence[str]):
        self.module_scanner = ModuleScanner(pkgs=pkgs, modules=modules)

    def build_component(self, config: Dict) -> object:
        """
        Build component instance based on the provided dictonary config.
        Supported keys for the config:
        - 'name' - class name in the modules of packages.
        - 'path' - directly specify the class path, based on PYTHONPATH, ignore 'name' if specified.
        - 'args' - arguments to initialize the component instance.
        - 'disabled' - if defined `'disabled': true`, will skip the buiding, useful for development or tuning.

        Args:
            config: dictionary config to define a component.

        Raises:
            ValueError: must provide `path` or `name` of class to build component.
            ValueError: can not find component class.

        """
        if not isinstance(config, dict):
            raise ValueError("config of component must be a dictionary.")

        if config.get("disabled") is True:
            # if marked as `disabled`, skip parsing
            return None

        class_args = config.get("args", {})
        class_path = self._get_class_path(config)
        return instantiate_class(class_path, **class_args)

    def _get_class_path(self, config):
        class_path = config.get("path", None)
        if class_path is None:
            class_name = config.get("name", None)
            if class_name is None:
                raise ValueError("must provide `path` or `name` of class to build component.")
            module_name = self.module_scanner.get_module_name(class_name)
            if module_name is None:
                raise ValueError(f"can not find component class '{class_name}'.")
            class_path = f"{module_name}.{class_name}"

        return class_path

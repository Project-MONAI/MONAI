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

import warnings
from typing import Any, Dict, List, Optional

from monai.apps.mmars.utils import search_configs_with_deps, update_configs_with_deps
from monai.utils.module import ClassScanner, instantiate_class


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

    Here we predefined several special marks to parse the config content:
    - "<XXX>": like "<name>" is the name of a class, to distinguish it with regular key "name" in the config content.
    now we have 4 keys: `<name>`, `<path>`, `<args>`, `<disabled>`.
    - "XXX#YYY": join nested config ids, like "transforms#5" is id name of the 6th transform in the transforms list.
    - "@XXX": use an instance as config item, like `"dataset": "@dataset"` uses `dataset` instance as the parameter.
    - "$XXX": execute the string after "$" as python code with `eval()` function, like "$@model.parameters()".

    Args:
        id: id name of current config component, for nested config items, use `#` to join ids.
            for list component, use index from `0` as id.
            for example: `transform`, `transform#5`, `transform#5#<args>#keys`, etc.
        config: config content of current component, can be a `dict`, `list`, `string`, `float`, `int`, etc.
        class_scanner: ClassScanner to help get the class name or path in the config and build instance.
        globals: to support executable string in the config, sometimes we need to provide the global variables
            which are referred in the executable string. for example: `globals={"monai": monai} will be useful
            for config `"collate_fn": "$monai.data.list_data_collate"`.

    """

    def __init__(self, id: str, config: Any, class_scanner: ClassScanner, globals: Optional[Dict] = None) -> None:
        self.id = id
        self.config = config
        self.class_scanner = class_scanner
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

    def build(self, config: Optional[Dict] = None) -> object:
        """
        Build component instance based on the provided dictonary config.
        Supported special keys for the config:
        - '<name>' - class name in the modules of packages.
        - '<path>' - directly specify the class path, based on PYTHONPATH, ignore '<name>' if specified.
        - '<args>' - arguments to initialize the component instance.
        - '<disabled>' - if defined `'<disabled>': True`, will skip the buiding, useful for development or tuning.

        Args:
            config: dictionary config that defines a component.

        Raises:
            ValueError: must provide `<path>` or `<name>` of class to build component.
            ValueError: can not find component class.

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

        class_args = config.get("<args>", {})
        class_path = self._get_class_path(config)
        return instantiate_class(class_path, **class_args)

    def _get_class_path(self, config):
        """
        Get the path of class specified in the config content.

        Args:
            config: dictionary config that defines a component.

        """
        class_path = config.get("<path>", None)
        if class_path is None:
            class_name = config.get("<name>", None)
            if class_name is None:
                raise ValueError("must provide `<path>` or `<name>` of class to build component.")
            module_name = self.class_scanner.get_class_module_name(class_name)
            if module_name is None:
                raise ValueError(f"can not find component class '{class_name}'.")
            class_path = f"{module_name}.{class_name}"

        return class_path

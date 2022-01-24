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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from monai.apps.mmars.config_resolver import ConfigComponent, ConfigResolver, ModuleScanner


class ConfigParser:
    """
    Parse dictionary format config and build components.

    Args:
        pkgs: the expected packages to scan modules and parse class names in the config.
        modules: the expected modules in the packages to scan for all the classes.
            for example, to parser "LoadImage" in config, `pkgs` can be ["monai"], `modules` can be ["transforms"].
        global_imports: pre-import packages as global variables to execute `eval` commands.
            for example, pre-import `monai`, then execute `eval("monai.data.list_data_collate")`.
        config: config content to parse.

    """

    def __init__(
        self,
        pkgs: Sequence[str],
        modules: Sequence[str],
        global_imports: Optional[Sequence[str]] = None,
        config: Optional[Any] = None,
    ):
        self.config = None
        if config is not None:
            self.set_config(config=config)
        self.module_scanner = ModuleScanner(pkgs=pkgs, modules=modules)
        self.global_imports = {}
        if global_imports is not None:
            for i in global_imports:
                self.global_imports[i] = importlib.import_module(i)
        self.config_resolver: Optional[ConfigResolver] = None
        self.resolved = False

    def _get_last_config_and_key(config: Union[Dict, List], path: str) -> Tuple[Union[Dict, List], str]:
        keys = path.split("#")
        for k in keys[:-1]:
            config = config[k] if isinstance(config, dict) else config[int(k)]
        key = keys[-1] if isinstance(config, dict) else int(keys[-1])
        return config, key

    def set_config(self, config: Any, path: Optional[str] = None):
        if isinstance(path, str):
            conf_, key = self._get_last_config_and_key(self.config, path)
            conf_[key] = config
        else:
            self.config = config
        self.resolved = False

    def get_config(self, path: Optional[str] = None):
        if isinstance(path, str):
            conf_, key = self._get_last_config_and_key(self.config, path)
            return conf_[key]
        return self.config

    def _do_scan(self, config, path: Optional[str] = None):
        if isinstance(config, dict):
            for k, v in config.items():
                sub_path = k if path is None else f"{path}#{k}"
                self._do_scan(config=v, path=sub_path)
        if isinstance(config, list):
            for i, v in enumerate(config):
                sub_path = i if path is None else f"{path}#{i}"
                self._do_scan(config=v, path=sub_path)
        if path is not None:
            self.config_resolver.update(
                ConfigComponent(id=path, config=config, module_scanner=self.module_scanner, globals=self.global_imports)
            )

    def resolve_config(self, resolve_all: bool = False):
        self.config_resolver = ConfigResolver()
        self._do_scan(config=self.config)

        if resolve_all:
            self.config_resolver.resolve_all()
        self.resolved = True

    def get_instance(self, id: str):
        if self.config_resolver is None or not self.resolved:
            self.resolve_config()
        return self.config_resolver.resolve_one_object(id=id)

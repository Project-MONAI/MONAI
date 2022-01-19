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

from typing import Any, Dict, Optional, Sequence
from config_resolver import ConfigComponent, ConfigResolver


class ConfigParser:
    """
    Parse dictionary format config and build components.

    Args:
        pkgs: the expected packages to scan.
        modules: the expected modules in the packages to scan.

    """

    def __init__(self, pkgs: Sequence[str], modules: Sequence[str], config: Optional[Dict] = None):
        self.pkgs = pkgs
        self.modules = modules
        self.config = {}
        if isinstance(config, dict):
            self.set_config(config=config)
        self.config_resolver: Optional[ConfigResolver] = None
        self.resolved = False

    def set_config(self, config: Any, path: Optional[str] = None):
        if isinstance(path, str):
            keys = path.split(".")
            config = self.config
            for k in keys[:-1]:
                config = config[k]
            config[keys[-1]] = config
        else:
            self.config = config
        self.resolved = False

    def get_config(self, config: Dict, path: Optional[str] = None):
        if isinstance(path, str):
            keys = path.split(".")
            config = self.config
            for k in keys[:-1]:
                config = config[k]
            return config[keys[-1]]
        return self.config

    def resolve_config(self, resolve_all: bool = False):
        self.config_resolver = ConfigResolver()
        for k, v in self.config.items():
            # only prepare the components, lazy instantiation
            # FIXME: only support "@" reference in top level config for now
            self.config_resolver.update(ConfigComponent(id=k, config=v, pkgs=self.pkgs, modules=self.modules))
        if resolve_all:
            self.config_resolver.resolve_all()
        self.resolved = True

    def get_instance(self, id: str):
        if self.config_resolver is None or not self.resolved:
            self.resolve_config()
        return self.config_resolver.resolve_one_object(id=id)

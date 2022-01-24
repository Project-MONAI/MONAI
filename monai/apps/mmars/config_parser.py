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
    Parse a nested config and build components.
    A typical usage is a config dictionary contains all the necessary components to define training workflow in JSON.
    For more details of the config format, please check :py:class:`monai.apps.ConfigComponent`.

    Args:
        pkgs: the expected packages to scan modules and parse class names in the config.
        modules: the expected modules in the packages to scan for all the classes.
            for example, to parser "LoadImage" in config, `pkgs` can be ["monai"], `modules` can be ["transforms"].
        global_imports: pre-import packages as global variables to execute the python `eval` commands.
            for example, pre-import `monai`, then execute `eval("monai.data.list_data_collate")`.
            default to `{"monai": "monai", "torch": "torch", "np": "numpy"}` as `numpy` and `torch`
            are MONAI mininum requirements.
        config: config content to parse.

    """

    def __init__(
        self,
        pkgs: Sequence[str],
        modules: Sequence[str],
        global_imports: Optional[Dict[str, str]] = {"monai": "monai", "torch": "torch", "np": "numpy"},
        config: Optional[Any] = None,
    ):
        self.config = None
        if config is not None:
            self.set_config(config=config)
        self.module_scanner = ModuleScanner(pkgs=pkgs, modules=modules)
        self.global_imports = {}
        if global_imports is not None:
            for k, v in global_imports.items():
                self.global_imports[k] = importlib.import_module(v)
        self.config_resolver: Optional[ConfigResolver] = None
        self.resolved = False

    def _get_last_config_and_key(self, config: Union[Dict, List], id: str) -> Tuple[Union[Dict, List], str]:
        """
        Utility to get the last config item and the id from the whole config content with nested id name.

        Args:
            config: the whole config content.
            id: nested id name to get the last item, joined by "#" mark, use index from 0 for list.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        keys = id.split("#")
        for k in keys[:-1]:
            config = config[k] if isinstance(config, dict) else config[int(k)]
        key = keys[-1] if isinstance(config, dict) else int(keys[-1])
        return config, key

    def set_config(self, config: Any, id: Optional[str] = None):
        """
        Set config content for the parser, if `id` provided, `config` will used to replace the config item with `id`.

        Args:
            config: target config content to set.
            id: nested id name to specify the target position, joined by "#" mark, use index from 0 for list.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        if isinstance(id, str):
            conf_, key = self._get_last_config_and_key(config=self.config, id=id)
            conf_[key] = config
        else:
            self.config = config
        self.resolved = False

    def get_config(self, id: Optional[str] = None):
        """
        Get config content from the parser, if `id` provided, get the config item with `id`.

        Args:
            id: nested id name to specify the expected position, joined by "#" mark, use index from 0 for list.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        if isinstance(id, str):
            conf_, key = self._get_last_config_and_key(config=self.config, id=id)
            return conf_[key]
        return self.config

    def _do_parse(self, config, id: Optional[str] = None):
        """
        Recursively parse the nested config content, add every config item as component to the resolver.
        For example, `{"preprocessing": [{"<name>": "LoadImage", "<args>": {"keys": "image"}}]}` is parsed as items:
        - `id="preprocessing", config=[{"<name>": "LoadImage", "<args>": {"keys": "image"}}]`
        - `id="preprocessing#0", config={"<name>": "LoadImage", "<args>": {"keys": "image"}}`
        - `id="preprocessing#0#<name>", config="LoadImage"`
        - `id="preprocessing#0#<args>", config={"keys": "image"}`
        - `id="preprocessing#0#<args>#keys", config="image"`

        Args:
            config: config content to parse.
            id: id name of current config item, nested ids are joined by "#" mark. defaults to None.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        if isinstance(config, dict):
            for k, v in config.items():
                sub_id = k if id is None else f"{id}#{k}"
                self._do_parse(config=v, id=sub_id)
        if isinstance(config, list):
            for i, v in enumerate(config):
                sub_id = i if id is None else f"{id}#{i}"
                self._do_parse(config=v, id=sub_id)
        if id is not None:
            self.config_resolver.add(
                ConfigComponent(id=id, config=config, module_scanner=self.module_scanner, globals=self.global_imports)
            )

    def parse_config(self, resolve_all: bool = False):
        """
        Parse the config content, add every config item as component to the resolver.

        Args:
            resolve_all: if True, resolve all the components and build instances directly.

        """
        self.config_resolver = ConfigResolver()
        self._do_parse(config=self.config)

        if resolve_all:
            self.config_resolver.resolve_all()
        self.resolved = True

    def get_resolved_config(self, id: str):
        """
        Get the resolved instance component, if not resolved, try to resolve it first.

        Args:
            id: id name of expected config component, nested ids are joined by "#" mark.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        if self.config_resolver is None or not self.resolved:
            self.parse_config()
        return self.config_resolver.get_resolved_config(id=id)

    def get_resolved_component(self, id: str):
        """
        Get the resolved config component, if not resolved, try to resolve it first.
        It can be used to modify the config again and support lazy instantiation.

        Args:
            id: id name of expected config component, nested ids are joined by "#" mark.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        if self.config_resolver is None or not self.resolved:
            self.parse_config()
        return self.config_resolver.get_resolved_component(id=id)

    def build(self, config: Dict):
        """
        Build a config to instance if no dependencies, usually used for lazy instantiation or ad-hoc build.

        Args:
            config: dictionary config content to build.

        """
        return ConfigComponent(
            id=None, config=config, module_scanner=self.module_scanner, globals=self.global_imports
            ).build()

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

import importlib
from typing import Any, Dict, List, Optional, Sequence, Union

from monai.apps.manifest.config_item import ConfigComponent, ConfigExpression, ConfigItem, ComponentLocator
from monai.apps.manifest.reference_resolver import ReferenceResolver


class ConfigParser:
    """
    Parse a nested config and build components.
    A typical usage is a config dictionary contains all the necessary components to define training workflow in JSON.
    For more details of the config format, please check :py:class:`monai.apps.ConfigComponent`.

    Args:
        excludes: if any string of the `excludes` exists in the full module name, don't import this module.
        global_imports: pre-import packages as global variables to execute the python `eval` commands.
            for example, pre-import `monai`, then execute `eval("monai.data.list_data_collate")`.
            default to `{"monai": "monai", "torch": "torch", "np": "numpy"}` as `numpy` and `torch`
            are MONAI mininum requirements.
        config: config content to parse.

    """

    def __init__(
        self,
        excludes: Optional[Union[Sequence[str], str]] = None,
        global_imports: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
    ):
        self.config = None
        if config is not None:
            self.set_config(config=config)
        self.locator = ComponentLocator(excludes=excludes)
        self.global_imports: Dict[str, Any] = {"monai": "monai", "torch": "torch", "np": "numpy"}
        if global_imports is not None:
            for k, v in global_imports.items():
                self.global_imports[k] = importlib.import_module(v)
        self.reference_resolver: Optional[ReferenceResolver] = None
        self.parsed = False

    def _get_last_config_and_key(self, config: Union[Dict, List], id: str):
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
        if isinstance(id, str) and isinstance(self.config, (dict, list)):
            conf_, key = self._get_last_config_and_key(config=self.config, id=id)
            conf_[key] = config
        else:
            self.config = config
        self.parsed = False

    def get_config(self, id: Optional[str] = None):
        """
        Get config content from the parser, if `id` provided, get the config item with `id`.

        Args:
            id: nested id name to specify the expected position, joined by "#" mark, use index from 0 for list.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        if isinstance(id, str) and isinstance(self.config, (dict, list)):
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
            if ConfigComponent.is_instantiable(config):
                self.reference_resolver.add(
                    ConfigComponent(config=config, id=id, locator=self.locator)
                )
            elif ConfigExpression.is_expression(config):
                self.reference_resolver.add(ConfigExpression(config=config, id=id, globals=self.global_imports))
            else:
                self.reference_resolver.add(ConfigItem(config=config, id=id))

    def parse_config(self):
        """
        Parse the config content, add every config item as component to the resolver.

        Args:
            resolve_all: if True, resolve all the components and build instances directly.

        """
        self.reference_resolver = ReferenceResolver()
        self._do_parse(config=self.config)
        self.parsed = True

    def get_resolved_content(self, id: str):
        """
        Get the resolved instance component, if not resolved, try to resolve it first.

        Args:
            id: id name of expected config component, nested ids are joined by "#" mark.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        if not self.parsed:
            self.parse_config()
        return self.reference_resolver.get_resolved_content(id=id)

    def get_config_item(self, id: str, resolve: bool = False):
        """
        Get the resolved config component, if not resolved, try to resolve it first.
        It can be used to modify the config again and support lazy instantiation.

        Args:
            id: id name of expected config component, nested ids are joined by "#" mark.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        if not self.parsed:
            self.parse_config()
        return self.reference_resolver.get_item(id=id, resolve=resolve)

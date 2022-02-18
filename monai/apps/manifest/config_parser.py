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
from typing import Any, Dict, Optional, Sequence, Union

from monai.apps.manifest.config_item import ConfigComponent, ConfigExpression, ConfigItem, ComponentLocator
from monai.apps.manifest.reference_resolver import ReferenceResolver


class ConfigParser:
    """
    Parse a config content, access or update the items of config content with unique ID.
    A typical usage is a config dictionary contains all the necessary components to define training workflow in JSON.
    For more details of the config format, please check :py:class:`monai.apps.ConfigComponent`.

    Args:
        config: config content to parse.
        excludes: when importing modules to instantiate components, if any string of the `excludes` exists
            in the full module name, don't import this module.
        globals: pre-import packages as global variables to evaluate the python `eval` expressions.
            for example, pre-import `monai`, then execute `eval("monai.data.list_data_collate")`.
            default to `{"monai": "monai", "torch": "torch", "np": "numpy"}` as `numpy` and `torch`
            are MONAI mininum requirements.
            if the value in global is string, will import it immediately.

    """

    def __init__(
        self,
        config: Optional[Any] = None,
        excludes: Optional[Union[Sequence[str], str]] = None,
        globals: Optional[Dict[str, Any]] = None,
    ):
        self.config = None
        if config is not None:
            self.set_config(config=config)

        self.globals: Dict[str, Any] = {}
        globals = {"monai": "monai", "torch": "torch", "np": "numpy"} if globals is None else globals
        if globals is not None:
            for k, v in globals.items():
                self.globals[k] = importlib.import_module(v) if isinstance(v, str) else v

        self.locator = ComponentLocator(excludes=excludes)
        self.reference_resolver: Optional[ReferenceResolver] = None
        # flag to identify the parsing status of current config content
        self.parsed = False

    def set_config(self, config: Any, id: Optional[str] = None):
        """
        Set config content for the parser, if `id` provided, `config` will replace the config item with `id`.

        Args:
            config: target config content to set.
            id: id name to specify the target position, joined by "#" mark for nested content, use index from 0 for list.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        if isinstance(id, str) and isinstance(self.config, (dict, list)):
            keys = id.split("#")
            # get the last second config item and replace it
            last_id = "#".join(keys[:-1])
            conf_ = self.get_config(id=last_id)
            conf_[keys[-1] if isinstance(conf_, dict) else int(keys[-1])] = config
        else:
            self.config = config
        # must totally parse again as the content is modified
        self.parsed = False

    def get_config(self, id: Optional[str] = None):
        """
        Get config content of current config, if `id` provided, get the config item with `id`.

        Args:
            id: nested id name to specify the expected position, joined by "#" mark, use index from 0 for list.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        config = self.config
        if isinstance(id, str) and len(id) > 0 and isinstance(config, (dict, list)):
            keys = id.split("#")
            for k in keys:
                config = config[k] if isinstance(config, dict) else config[int(k)]
        return config

    def _do_parse(self, config, id: Optional[str] = None):
        """
        Recursively parse the nested config content, add every config item to the resolver.
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
        # add config items to the resolver
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
        Parse the config content, add every config item to the resolver and mark as `parsed`.

        """
        self.reference_resolver = ReferenceResolver()
        self._do_parse(config=self.config)
        self.parsed = True

    def get_resolved_content(self, id: str):
        """
        Get the resolved result of config items with specified id, if not resolved, try to resolve it first.
        If the config item is instantiable, the resolved result is the instance.
        If the config item is an expression, the resolved result is output of when evaluating the expression.
        Otherwise, the resolved result is the updated config content of the config item.

        Args:
            id: id name of expected config item, nested ids are joined by "#" mark.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        if not self.parsed:
            self.parse_config()
        return self.reference_resolver.get_resolved_content(id=id)

    def get_config_item(self, id: str, resolve: bool = False):
        """
        Get the parsed config item, if `resolve=True` and not resolved, try to resolve it first.
        It can be used to modify the config in other program and support lazy instantiation.

        Args:
            id: id name of expected config component, nested ids are joined by "#" mark.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        if not self.parsed:
            self.parse_config()
        return self.reference_resolver.get_item(id=id, resolve=resolve)

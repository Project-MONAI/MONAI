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
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Union

from monai.apps.manifest.config_item import ComponentLocator, ConfigComponent, ConfigExpression, ConfigItem
from monai.apps.manifest.reference_resolver import ReferenceResolver


class ConfigParser:
    """
    Parse a config source, access or update the content of the config source with unique ID.
    A typical usage is a config dictionary contains all the necessary information to define training workflow in JSON.
    For more details of the config format, please check :py:class:`monai.apps.ConfigItem`.

    It can recursively parse the config source, treat every item as a `ConfigItem` with unique ID, the ID is joined
    by "#" mark for nested items. For example:
    The config source `{"preprocessing": [{"<name>": "LoadImage", "<args>": {"keys": "image"}}]}` is parsed as items:
    - `id="preprocessing", config=[{"<name>": "LoadImage", "<args>": {"keys": "image"}}]`
    - `id="preprocessing#0", config={"<name>": "LoadImage", "<args>": {"keys": "image"}}`
    - `id="preprocessing#0#<name>", config="LoadImage"`
    - `id="preprocessing#0#<args>", config={"keys": "image"}`
    - `id="preprocessing#0#<args>#keys", config="image"`

    A typical workflow of config parsing is as follows:

    - Initialize `ConfigParser` with the `config` source.
    - Call ``get_parsed_content()`` to get expected component with `id`, which will be automatically parsed.

    .. code-block:: python

        config = {
            "preprocessing": {"<name>": "LoadImage"},
            "net": {"<name>": "UNet", "<args>": ...},
            "trainer": {"<name>": "SupervisedTrainer", "<args>": {"network": "@net", ...}},
        }
        parser = ConfigParser(config=config)
        trainer = parser.get_parsed_content(id="trainer")
        trainer.run()

    It's also flexible to modify config source at runtime and parse again:

    .. code-block:: python

        parser = ConfigParser(...)
        config = parser.get_config()
        config["processing"][2]["<args>"]["interp_order"] = "bilinear"
        parser.parse_config()
        trainer = parser.get_parsed_content(id="trainer")
        trainer.run()

    Args:
        config: input config source to parse.
        id: specified ID name for the config sources.
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
        config: Any,
        excludes: Optional[Union[Sequence[str], str]] = None,
        globals: Optional[Dict[str, Any]] = None,
    ):
        self.config = None
        self.globals: Dict[str, Any] = {}
        globals = {"monai": "monai", "torch": "torch", "np": "numpy"} if globals is None else globals
        if globals is not None:
            for k, v in globals.items():
                self.globals[k] = importlib.import_module(v) if isinstance(v, str) else v

        self.locator = ComponentLocator(excludes=excludes)
        self.reference_resolver = ReferenceResolver()
        self.update_config({"": config})

    def update_config(self, content: Dict[str, Any]):
        """
        Update config source for the parser, every `key` and `value` in the `content` is corresponding to
        the target `id` position and new config value in order.
        Nested config id is joined by "#" mark, use index from 0 for list.
        For example: "transforms#5", "transforms#5#<args>#keys", etc.
        If `id` is `""`, replace `self.config`.

        """
        for id, config in content.items():
            if id != "" and isinstance(self.config, (dict, list)):
                keys = id.split("#")
                # get the last second config item and replace it
                last_id = "#".join(keys[:-1])
                conf_ = self.get_config(id=last_id)
                conf_[keys[-1] if isinstance(conf_, dict) else int(keys[-1])] = config
            else:
                self.config = config
        # must totally parse again as the content is modified
        self.parse_config()

    def get_config(self, id: str = ""):
        """
        Get config source in the parser, if `id` provided, get the config item with `id`.

        Args:
            id: id name to specify the expected position, nested config is joined by "#" mark, use index from 0 for list.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.
                default to get all the config source data.

        """
        config = self.config
        if id != "" and isinstance(config, (dict, list)):
            keys = id.split("#")
            for k in keys:
                config = config[k] if isinstance(config, dict) else config[int(k)]
        return config

    def _do_parse(self, config, id: str = ""):
        """
        Recursively parse the nested data in config source, add every config item to the resolver.

        Args:
            config: config source to parse.
            id: id name of current config item, nested ids are joined by "#" mark. defaults to None.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.
                default to empty string.

        """
        if isinstance(config, (dict, list)):
            subs = enumerate(config) if isinstance(config, list) else config.items()
            for k, v in subs:
                sub_id = f"{id}#{k}" if id != "" else k
                self._do_parse(config=v, id=sub_id)

        if id != "":
            # copy every config item to make them independent and add them to the resolver
            item_conf = deepcopy(config)
            if ConfigComponent.is_instantiable(item_conf):
                self.reference_resolver.add_item(ConfigComponent(config=item_conf, id=id, locator=self.locator))
            elif ConfigExpression.is_expression(item_conf):
                self.reference_resolver.add_item(ConfigExpression(config=item_conf, id=id, globals=self.globals))
            else:
                self.reference_resolver.add_item(ConfigItem(config=item_conf, id=id))

    def parse_config(self):
        """
        Parse the config source, add every config item to the resolver.

        """
        self.reference_resolver.reset()
        self._do_parse(config=self.config)

    def get_parsed_content(self, id: str):
        """
        Get the parsed result of config item with specified id, if having references not resolved,
        try to resolve it first.

        If the config item is `ConfigComponent`, the parsed result is the instance.
        If the config item is `ConfigExpression`, the parsed result is output of evaluating the expression.
        Otherwise, the parsed result is the updated `self.config` data of `ConfigItem`.

        Args:
            id: id name of expected config item, nested ids are joined by "#" mark.
                for example: "transforms#5", "transforms#5#<args>#keys", etc.

        """
        return self.reference_resolver.get_resolved_content(id=id)

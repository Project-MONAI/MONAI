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

from monai.bundle.config_item import ComponentLocator, ConfigComponent, ConfigExpression, ConfigItem
from monai.bundle.reference_resolver import ReferenceResolver


class ConfigParser:
    """
    The primary configuration parser. It traverses a structured config (in the form of nested Python dict or list),
    creates ``ConfigItem``, and assign unique IDs according to the structures.

    This class provides convenient access to the set of ``ConfigItem`` of the config by ID.
    A typical workflow of config parsing is as follows:

        - Initialize ``ConfigParser`` with the ``config`` source.
        - Call ``get_parsed_content()`` to get expected component with `id`.

    .. code-block:: python

        from monai.apps import ConfigParser

        config = {
            "my_dims": 2,
            "dims_1": "$@my_dims + 1",
            "my_xform": {"<name>": "LoadImage"},
            "my_net": {"<name>": "BasicUNet",
                       "<args>": {"spatial_dims": "@dims_1", "in_channels": 1, "out_channels": 4}},
            "trainer": {"<name>": "SupervisedTrainer",
                        "<args>": {"network": "@my_net", "preprocessing": "@my_xform"}}
        }
        # in the example $@my_dims + 1 is an expression, which adds 1 to the value of @my_dims
        parser = ConfigParser(config)

        # get/set configuration content, the set method should happen before calling parse()
        print(parser["my_net"]["<args>"]["in_channels"])  # original input channels 1
        parser["my_net"]["<args>"]["in_channels"] = 4  # change input channels to 4
        print(parser["my_net"]["<args>"]["in_channels"])

        # instantiate the network component
        parser.parse(True)
        net = parser.get_parsed_content("my_net", instantiate=True)
        print(net)

        # also support to get the configuration content of parsed `ConfigItem`
        trainer = parser.get_parsed_content("trainer", instantiate=False)
        print(trainer)

    Args:
        config: input config source to parse.
        excludes: when importing modules to instantiate components,
            excluding components from modules specified in ``excludes``.
        globals: pre-import packages as global variables to ``ConfigExpression``,
            so that expressions, for example, ``"$monai.data.list_data_collate"`` can use ``monai`` modules.
            The current supported globals and alias names are
            ``{"monai": "monai", "torch": "torch", "np": "numpy", "numpy": "numpy"}``.
            These are MONAI's minimal dependencies.

    See also:

        - :py:class:`monai.apps.ConfigItem`

    """

    def __init__(
        self,
        config: Any,
        excludes: Optional[Union[Sequence[str], str]] = None,
        globals: Optional[Dict[str, Any]] = None,
    ):
        self.config = None
        self.globals: Dict[str, Any] = {}
        globals = {"monai": "monai", "torch": "torch", "np": "numpy", "numpy": "numpy"} if globals is None else globals
        if globals is not None:
            for k, v in globals.items():
                self.globals[k] = importlib.import_module(v) if isinstance(v, str) else v

        self.locator = ComponentLocator(excludes=excludes)
        self.ref_resolver = ReferenceResolver()
        self.set(config=config)

    def __repr__(self):
        return f"{self.config}"

    def __getitem__(self, id: Union[str, int]):
        """
        Get the config by id.

        Args:
            id: id of the ``ConfigItem``, ``"#"`` in id are interpreted as special characters to
                go one level further into the nested structures.
                Use digits indexing from "0" for list or other strings for dict.
                For example: ``"xform#5"``, ``"net#<args>#channels"``. ``""`` indicates the entire ``self.config``.

        """
        if id == "":
            return self.config
        config = self.config
        for k in str(id).split(self.ref_resolver.sep):
            if not isinstance(config, (dict, list)):
                raise ValueError(f"config must be dict or list for key `{k}`, but got {type(config)}: {config}.")
            indexing = k if isinstance(config, dict) else int(k)
            config = config[indexing]
        return config

    def __setitem__(self, id: Union[str, int], config: Any):
        """
        Set config by ``id``.  Note that this method should be used before ``parse()`` or ``get_parsed_content()``
        to ensure the updates are included in the parsed content.

        Args:
            id: id of the ``ConfigItem``, ``"#"`` in id are interpreted as special characters to
                go one level further into the nested structures.
                Use digits indexing from "0" for list or other strings for dict.
                For example: ``"xform#5"``, ``"net#<args>#channels"``. ``""`` indicates the entire ``self.config``.
            config: config to set at location ``id``.

        """
        if id == "":
            self.config = config
            self.ref_resolver.reset()
            return
        keys = str(id).split(self.ref_resolver.sep)
        # get the last parent level config item and replace it
        last_id = self.ref_resolver.sep.join(keys[:-1])
        conf_ = self[last_id]
        indexing = keys[-1] if isinstance(conf_, dict) else int(keys[-1])
        conf_[indexing] = config
        self.ref_resolver.reset()
        return

    def get(self, id: str = "", default: Optional[Any] = None):
        """
        Get the config by id.

        Args:
            id: id to specify the expected position. See also :py:meth:`__getitem__`.
            default: default value to return if the specified ``id`` is invalid.

        """
        try:
            return self[id]
        except KeyError:
            return default

    def set(self, config: Any, id: str = ""):
        """
        Set config by ``id``. See also :py:meth:`__setitem__`.

        """
        self[id] = config

    def _do_parse(self, config, id: str = ""):
        """
        Recursively parse the nested data in config source, add every item as `ConfigItem` to the resolver.

        Args:
            config: config source to parse.
            id: id of the ``ConfigItem``, ``"#"`` in id are interpreted as special characters to
                go one level further into the nested structures.
                Use digits indexing from "0" for list or other strings for dict.
                For example: ``"xform#5"``, ``"net#<args>#channels"``. ``""`` indicates the entire ``self.config``.

        """
        if isinstance(config, (dict, list)):
            subs = enumerate(config) if isinstance(config, list) else config.items()
            for k, v in subs:
                sub_id = f"{id}{self.ref_resolver.sep}{k}" if id != "" else k
                self._do_parse(config=v, id=sub_id)

        # copy every config item to make them independent and add them to the resolver
        item_conf = deepcopy(config)
        if ConfigComponent.is_instantiable(item_conf):
            self.ref_resolver.add_item(ConfigComponent(config=item_conf, id=id, locator=self.locator))
        elif ConfigExpression.is_expression(item_conf):
            self.ref_resolver.add_item(ConfigExpression(config=item_conf, id=id, globals=self.globals))
        else:
            self.ref_resolver.add_item(ConfigItem(config=item_conf, id=id))

    def parse(self, reset: bool = True):
        """
        Recursively parse the config source, add every item as ``ConfigItem`` to the resolver.

        Args:
            reset: whether to reset the ``reference_resolver`` before parsing. Defaults to `True`.

        """
        if reset:
            self.ref_resolver.reset()
        self._do_parse(config=self.config)

    def get_parsed_content(self, id: str = "", **kwargs):
        """
        Get the parsed result of ``ConfigItem`` with the specified ``id``.

            - If the item is ``ConfigComponent`` and ``instantiate=True``, the result is the instance.
            - If the item is ``ConfigExpression`` and ``eval_expr=True``, the result is the evaluated output.
            - Else, the result is the configuration content of `ConfigItem`.

        Args:
            id: id of the ``ConfigItem``, ``"#"`` in id are interpreted as special characters to
                go one level further into the nested structures.
                Use digits indexing from "0" for list or other strings for dict.
                For example: ``"xform#5"``, ``"net#<args>#channels"``. ``""`` indicates the entire ``self.config``.
            kwargs: additional keyword arguments to be passed to ``_resolve_one_item``.
                Currently support ``reset`` (for parse), ``instantiate`` and ``eval_expr``. All defaulting to True.

        """
        if not self.ref_resolver.is_resolved():
            # not parsed the config source yet, parse it
            self.parse(kwargs.get("reset", True))
        return self.ref_resolver.get_resolved_content(id=id, **kwargs)

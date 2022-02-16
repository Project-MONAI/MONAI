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

from monai.apps.mmars.config_item import ConfigComponent, ConfigItem, ComponentLocator
from monai.apps.mmars.utils import is_instantiable

class ConfigResolver:
    """
    Utility class to resolve the dependencies between config components and build instance for specified `id`.

    Args:
        components: config components to resolve, if None, can also `add()` component in runtime.

    """

    def __init__(self, components: Optional[Dict[str, ConfigComponent]] = None):
        self.resolved_configs: Dict[str, str] = {}
        self.resolved_components: Dict[str, Any] = {}
        self.components = {} if components is None else components

    def add(self, component: ConfigComponent):
        """
        Add a component to the resolution graph.

        Args:
            component: a config component to resolve.

        """
        id = component.get_id()
        if id in self.components:
            raise ValueError(f"id '{id}' is already added.")
        self.components[id] = component

    def _resolve_one_component(self, id: str, instantiate: bool = True, waiting_list: Optional[List[str]] = None):
        """
        Resolve one component with specified id name.
        If has unresolved dependencies, recursively resolve the dependencies first.

        Args:
            id: id name of expected component to resolve.
            instantiate: after resolving all the dependencies, whether to build instance.
                if False, can support lazy instantiation with the resolved config later.
                default to `True`.
            waiting_list: list of components wait to resolve dependencies. it's used to detect circular dependencies
                when resolving dependencies like: `{"name": "A", "dep": "@B"}` and `{"name": "B", "dep": "@A"}`.

        """
        if waiting_list is None:
            waiting_list = []
        waiting_list.append(id)
        com = self.components[id]
        dep_ids = com.get_dependent_ids()
        # if current component has dependency already in the waiting list, that's circular dependencies
        for d in dep_ids:
            if d in waiting_list:
                raise ValueError(f"detected circular dependencies for id='{d}' in the config content.")

        deps = {}
        if len(dep_ids) > 0:
            # # check whether the component has any unresolved deps
            for comp_id in dep_ids:
                if comp_id not in self.resolved_components:
                    # this dependent component is not resolved
                    if comp_id not in self.components:
                        raise RuntimeError(f"the dependent component `{comp_id}` is not in config.")
                    # resolve the dependency first
                    self._resolve_one_component(id=comp_id, instantiate=True, waiting_list=waiting_list)
                deps[comp_id] = self.resolved_components[comp_id]
            # all dependent components are resolved already
        updated_config = com.get_updated_config(deps)
        resolved_com = None

        if instantiate:
            resolved_com = com.build(updated_config)
            self.resolved_configs[id] = updated_config
            self.resolved_components[id] = resolved_com

        return updated_config, resolved_com

    def resolve_all(self):
        """
        Resolve all the components and build instances.

        """
        for k in self.components.keys():
            self._resolve_one_component(id=k, instantiate=True)

    def get_resolved_component(self, id: str):
        """
        Get the resolved instance component with specified id name.
        If not resolved, try to resolve it first.

        Args:
            id: id name of the expected component.

        """
        if id not in self.resolved_components:
            self._resolve_one_component(id=id, instantiate=True)
        return self.resolved_components[id]

    def get_resolved_config(self, id: str):
        """
        Get the resolved config component with specified id name, then can be used for lazy instantiation.
        If not resolved, try to resolve it with `instantiation=False` first.

        Args:
            id: id name of the expected config component.

        """
        if id not in self.resolved_configs:
            config, _ = self._resolve_one_component(id=id, instantiate=False)
        else:
            config = self.resolved_configs[id]
        return config


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
        self.config_resolver: ConfigResolver = ConfigResolver()
        self.resolved = False

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
        self.resolved = False

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
            if is_instantiable(config):
                self.config_resolver.add(
                    ConfigComponent(id=id, config=config, locator=self.locator, globals=self.global_imports)
                )
            else:
                self.config_resolver.add(ConfigItem(id=id, config=config, globals=self.global_imports))

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

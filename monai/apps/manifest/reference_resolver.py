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

import re
import warnings
from typing import Any, Dict, Optional, Sequence, Set

from monai.apps.manifest.config_item import ConfigComponent, ConfigExpression, ConfigItem


class ReferenceResolver:
    """
    Utility class to manage a set of ``ConfigItem`` and resolve the references between them.

    This class maintains a set of ``ConfigItem`` objects and their associated IDs.
    The IDs must be unique within this set. A string in ``ConfigItem``
    starting with ``@`` will be treated as a reference to other ``ConfigItem`` objects by ID.
    Since ``ConfigItem`` may have a nested dictionary or list structure,
    the reference string may also contain a ``#`` character to refer to a substructure by
    key indexing for a dictionary or integer indexing for a list.

    In this class, resolving references is essentially substitution of the reference strings with the
    corresponding python objects. A typical workflow of resolving references is as follows:

        - Add multiple ``ConfigItem`` objects to the ``ReferenceResolver`` by ``add_item()``.
        - Call ``get_resolved_content()`` to automatically resolve the references. This is done (recursively) by:
            - Convert the items to objects, for those do not have references to other items.
                - If it is instantiable, instantiate it and cache the class instance in ``resolved_content``.
                - If it is an expression, evaluate it and save the value in ``resolved_content``.
            - Substitute the reference strings with the corresponding objects.

    Args:
        items: ``ConfigItem``s to resolve, this could be added later with ``add_item()``.

    """

    def __init__(self, items: Optional[Sequence[ConfigItem]] = None):
        # save the items in a dictionary with the `ConfigItem.id` as key
        self.items: Dict[str, Any] = {} if items is None else {i.get_id(): i for i in items}
        self.resolved_content: Dict[str, Any] = {}

    def reset(self):
        """
        Clear all the added `ConfigItem` and all the resolved content.

        """
        self.items = {}
        self.resolved_content = {}

    def add_item(self, item: ConfigItem):
        """
        Add a ``ConfigItem`` to the resolver.

        Args:
            item: a ``ConfigItem``.

        """
        id = item.get_id()
        if id in self.items:
            warnings.warn(f"id '{id}' is already added.")
            return
        self.items[id] = item

    def get_item(self, id: str, resolve: bool = False):
        """
        Get the ``ConfigItem`` by id.

        If ``resolve=True``, the returned item will be resolved, that is,
        all the reference strings are substituted by the corresponding ``ConfigItem`` objects.

        Args:
            id: id of the expected config item.
            resolve: whether to resolve the item if it is not resolved, default to False.

        """
        if resolve and id not in self.resolved_content:
            self._resolve_one_item(id=id)
        return self.items.get(id)

    def _resolve_one_item(self, id: str, waiting_list: Optional[Set[str]] = None):
        """
        Resolve one ``ConfigItem`` of ``id``, cache the resolved result in ``resolved_content``.
        If it has unresolved references, recursively resolve the referring items first.

        Args:
            id: id name of ``ConfigItem`` to be resolved.
            waiting_list: set of ids pending to be resolved.
                It's used to detect circular references such as:
                `{"name": "A", "dep": "@B"}` and `{"name": "B", "dep": "@A"}`.

        """
        item = self.items[id]  # if invalid id name, raise KeyError
        item_config = item.get_config()

        if waiting_list is None:
            waiting_list = set()
        waiting_list.add(id)

        ref_ids = self.find_refs_in_config(config=item_config, id=id)

        # if current item has reference already in the waiting list, that's circular references
        for d in ref_ids:
            if d in waiting_list:
                raise ValueError(f"detected circular references for id='{d}' in the config content.")

        # # check whether the component has any unresolved references
        for d in ref_ids:
            if d not in self.resolved_content:
                # this referring item is not resolved
                if d not in self.items:
                    raise ValueError(f"the referring item `{d}` is not defined in config.")
                # recursively resolve the reference first
                self._resolve_one_item(id=d, waiting_list=waiting_list)

        # all references are resolved, then try to resolve current config item
        new_config = self.update_config_with_refs(config=item_config, id=id, refs=self.resolved_content)
        item.update_config(config=new_config)
        # save the resolved result into `resolved_content` to recursively resolve others
        if isinstance(item, ConfigComponent):
            self.resolved_content[id] = item.instantiate()
        elif isinstance(item, ConfigExpression):
            self.resolved_content[id] = item.evaluate(locals={"refs": self.resolved_content})
        else:
            self.resolved_content[id] = new_config

    def get_resolved_content(self, id: str):
        """
        Get the resolved ``ConfigItem`` by id. If there are unresolved references, try to resolve them first.

        Args:
            id: id name of the expected item.

        """
        if id not in self.resolved_content:
            self._resolve_one_item(id=id)
        return self.resolved_content[id]

    @staticmethod
    def match_refs_pattern(value: str) -> Set[str]:
        """
        Match regular expression for the input string to find the references.
        The reference string starts with ``"@"``, like: ``"@XXX#YYY#ZZZ"``.

        Args:
            value: input value to match regular expression.

        """
        refs: Set[str] = set()
        # regular expression pattern to match "@XXX" or "@XXX#YYY"
        result = re.compile(r"@\w*[\#\w]*").findall(value)
        for item in result:
            if ConfigExpression.is_expression(value) or value == item:
                # only check when string starts with "$" or the whole content is "@XXX"
                refs.add(item[1:])
        return refs

    @staticmethod
    def update_refs_pattern(value: str, refs: Dict) -> str:
        """
        Match regular expression for the input string to update content with the references.
        The reference part starts with ``"@"``, like: ``"@XXX#YYY#ZZZ"``.
        References dictionary must contain the referring IDs as keys.

        Args:
            value: input value to match regular expression.
            refs: all the referring components with ids as keys, default to `None`.

        """
        # regular expression pattern to match "@XXX" or "@XXX#YYY"
        result = re.compile(r"@\w*[\#\w]*").findall(value)
        for item in result:
            ref_id = item[1:]
            if ref_id not in refs:
                raise KeyError(f"can not find expected ID '{ref_id}' in the references.")
            if ConfigExpression.is_expression(value):
                # replace with local code, will be used in the `evaluate` logic with `locals={"refs": ...}`
                value = value.replace(item, f"refs['{ref_id}']")
            elif value == item:
                # the whole content is "@XXX", it will avoid the case that regular string contains "@"
                value = refs[ref_id]
        return value

    @staticmethod
    def find_refs_in_config(config, id: str, refs: Optional[Set[str]] = None) -> Set[str]:
        """
        Recursively search all the content of input config item to get the ids of references.
        References mean: the IDs of other config items (``"@XXX"`` in this config item), or the
        sub-item in the config is `instantiable`, or the sub-item in the config is `expression`.
        For `dict` and `list`, recursively check the sub-items.

        Args:
            config: input config content to search.
            id: ID name for the input config item.
            refs: list of the ID name of found references, default to `None`.

        """
        refs_: Set[str] = refs or set()
        if isinstance(config, str):
            return refs_.union(ReferenceResolver.match_refs_pattern(value=config))
        if not isinstance(config, (list, dict)):
            return refs_
        for k, v in config.items() if isinstance(config, dict) else enumerate(config):
            sub_id = f"{id}#{k}" if id != "" else f"{k}"
            if ConfigComponent.is_instantiable(v) or ConfigExpression.is_expression(v):
                refs_.add(sub_id)
            refs_ = ReferenceResolver.find_refs_in_config(v, sub_id, refs_)
        return refs_

    @staticmethod
    def update_config_with_refs(config, id: str, refs: Optional[Dict] = None):
        """
        With all the references in ``refs``, update the input config content with references
        and return the new config.

        Args:
            config: input config content to update.
            id: ID name for the input config.
            refs: all the referring content with ids, default to `None`.

        """
        refs_: Dict = refs or {}
        if isinstance(config, str):
            return ReferenceResolver.update_refs_pattern(config, refs_)
        if not isinstance(config, (list, dict)):
            return config
        ret = type(config)()
        for idx, v in config.items() if isinstance(config, dict) else enumerate(config):
            sub_id = f"{id}#{idx}" if id != "" else f"{idx}"
            if ConfigComponent.is_instantiable(v) or ConfigExpression.is_expression(v):
                updated = ReferenceResolver.update_config_with_refs(v, sub_id, refs_)
            else:
                updated = ReferenceResolver.update_config_with_refs(v, sub_id, refs_)
            ret.update({idx: updated}) if isinstance(ret, dict) else ret.append(updated)
        return ret

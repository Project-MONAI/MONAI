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
from typing import Dict, List, Optional, Sequence, Set, Union

from monai.apps.manifest.config_item import ConfigComponent, ConfigExpression, ConfigItem


class ReferenceResolver:
    """
    Utility class to manage config items and resolve the references between them.

    There are 3 kinds of `references` in the config content:
    - The IDs of other config items used as "@XXX" in this config item, for example:
    config item with ID="A" is a list `[1, 2, 3]`, another config item "B" can be `"args": {"input_list": "@A"}`.
    Then it means A is one reference of B.
    - If sub-item in the config is `instantiable`, treat it as reference because must instantiate the sub-item
    before using this config.
    - If sub-item in the config is `expression`, also treat it as reference because must evaluate the expression
    before using this config.

    The typical usage of the APIs:
    - Automatically search the content of specified config item and find out all the references.
    - Recursively resolve the references of this config item and update them in the config content.
    - If this config item is instantiable, try to instantiate it and save the instance in the `resolved_content`.
    If this config item is an expression, try to evaluate it and save the result in the `resolved_content`.
    Otherwise, save the updated config content in the `resolved_content`.

    Args:
        items: config items to resolve, if None, can also `add()` component in runtime.

    """

    def __init__(self, items: Optional[Sequence[ConfigItem]] = None):
        # save the items in a dictionary with the `id` as key
        self.items = {} if items is None else {i.get_id(): i for i in items}
        self.resolved_content = {}

    def add_item(self, item: ConfigItem):
        """
        Add a config item to the resolver.

        Args:
            item: a config item to resolve.

        """
        id = item.get_id()
        if id in self.items:
            warnings.warn(f"id '{id}' is already added.")
            return
        self.items[id] = item

    def get_item(self, id: str, resolve: bool = False):
        """
        Get the config item with specified id name, then can be used for lazy instantiation, etc.
        If `resolve=True` and the item is not resolved, try to resolve it first, then it will have
        no reference in the config content.

        Args:
            id: id name of the expected config item.

        """
        if resolve and id not in self.resolved_content:
            self._resolve_one_item(id=id)
        return self.items.get(id)

    def _resolve_one_item(self, id: str, waiting_list: Optional[Set[str]] = None):
        """
        Resolve one config item with specified id name, save the resolved result in `resolved_content`.
        If it has unresolved references, recursively resolve the referring items first.

        Args:
            id: id name of expected config item to resolve.
            waiting_list: list of the ids of items wait to resolve references.
                it's used to detect circular references when resolving references like:
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

        if len(ref_ids) > 0:
            # # check whether the component has any unresolved references
            for ref_id in ref_ids:
                if ref_id not in self.resolved_content:
                    # this referring item is not resolved
                    if ref_id not in self.items:
                        raise ValueError(f"the referring item `{ref_id}` is not defined in config.")
                    # recursively resolve the reference first
                    self._resolve_one_item(id=ref_id, waiting_list=waiting_list)

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

    def resolve_all(self):
        """
        Resolve the references for all the config items.

        """
        for k in self.items:
            self._resolve_one_item(id=k)

    def get_resolved_content(self, id: str):
        """
        Get the resolved content with specified id name.
        If not resolved yet, try to resolve it first.

        Args:
            id: id name of the expected item.

        """
        if id not in self.resolved_content:
            self._resolve_one_item(id=id)
        return self.resolved_content[id]

    @staticmethod
    def match_refs_pattern(value: str) -> List[str]:
        """
        Match regular expression for the input string to find the references.
        The reference part starts with "@", like: "@XXX#YYY#ZZZ".

        Args:
            value: input value to match regular expression.

        """
        refs: List[str] = []
        # regular expression pattern to match "@XXX" or "@XXX#YYY"
        result = re.compile(r"@\w*[\#\w]*").findall(value)
        for item in result:
            if ConfigExpression.is_expression(value) or value == item:
                # only check when string starts with "$" or the whole content is "@XXX"
                ref_obj_id = item[1:]
                if ref_obj_id not in refs:
                    refs.append(ref_obj_id)
        return refs

    @staticmethod
    def update_refs_pattern(value: str, refs: Dict) -> str:
        """
        Match regular expression for the input string to update content with the references.
        The reference part starts with "@", like: "@XXX#YYY#ZZZ".
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
    def find_refs_in_config(
        config: Union[Dict, List, str], id: Optional[str] = None, refs: Optional[List[str]] = None
    ) -> List[str]:
        """
        Recursively search all the content of input config item to get the ids of references.
        References mean: the IDs of other config items used as "@XXX" in this config item, or the
        sub-item in the config is `instantiable`, or the sub-item in the config is `expression`.
        For `dict` and `list`, recursively check the sub-items.

        Args:
            config: input config content to search.
            id: ID name for the input config item, default to `None`.
            refs: list of the ID name of found references, default to `None`.

        """
        refs_: List[str] = [] if refs is None else refs
        if isinstance(config, str):
            refs_ += ReferenceResolver.match_refs_pattern(value=config)

        if isinstance(config, (list, dict)):
            subs = enumerate(config) if isinstance(config, list) else config.items()
            for k, v in subs:
                sub_id = f"{id}#{k}" if id is not None else f"{k}"
                if ConfigComponent.is_instantiable(v) or ConfigExpression.is_expression(v):
                    refs_.append(sub_id)
                refs_ = ReferenceResolver.find_refs_in_config(v, sub_id, refs_)
        return refs_

    @staticmethod
    def update_config_with_refs(config: Union[Dict, List, str], id: Optional[str] = None, refs: Optional[Dict] = None):
        """
        With all the references in `refs`, update the input config content with references
        and return the new config.

        Args:
            config: input config content to update.
            id: ID name for the input config, default to `None`.
            refs: all the referring content with ids, default to `None`.

        """
        refs_: Dict = {} if refs is None else refs
        if isinstance(config, str):
            config = ReferenceResolver.update_refs_pattern(config, refs)
        if isinstance(config, list):
            # all the items in the list should be replaced with the references
            ret_list: List = []
            for i, v in enumerate(config):
                sub_id = f"{id}#{i}" if id is not None else f"{i}"
                if ConfigComponent.is_instantiable(v) or ConfigExpression.is_expression(v):
                    ret_list.append(refs_[sub_id])
                else:
                    ret_list.append(ReferenceResolver.update_config_with_refs(v, sub_id, refs_))
            return ret_list
        if isinstance(config, dict):
            # all the items in the dict should be replaced with the references
            ret_dict: Dict = {}
            for k, v in config.items():
                sub_id = f"{id}#{k}" if id is not None else f"{k}"
                if ConfigComponent.is_instantiable(v) or ConfigExpression.is_expression(v):
                    ret_dict[k] = refs_[sub_id]
                else:
                    ret_dict[k] = ReferenceResolver.update_config_with_refs(v, sub_id, refs_)
            return ret_dict
        return config

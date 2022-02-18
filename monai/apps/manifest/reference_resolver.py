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
from typing import Dict, List, Optional, Union
import warnings

from monai.apps.manifest.config_item import ConfigComponent, ConfigExpression, ConfigItem


class ReferenceResolver:
    """
    Utility class to resolve the references between config items.

    Args:
        components: config components to resolve, if None, can also `add()` component in runtime.

    """

    def __init__(self, items: Optional[Dict[str, ConfigItem]] = None):
        self.items = {} if items is None else items
        self.resolved_content = {}

    def add(self, item: ConfigItem):
        """
        Add a config item to the resolution graph.

        Args:
            item: a config item to resolve.

        """
        id = item.get_id()
        if id in self.items:
            warnings.warn(f"id '{id}' is already added.")
            return
        self.items[id] = item

    def resolve_one_item(self, id: str, waiting_list: Optional[List[str]] = None):
        """
        Resolve one item with specified id name.
        If has unresolved references, recursively resolve the references first.

        Args:
            id: id name of expected item to resolve.
            waiting_list: list of items wait to resolve references. it's used to detect circular references.
                when resolving references like: `{"name": "A", "dep": "@B"}` and `{"name": "B", "dep": "@A"}`.

        """
        if waiting_list is None:
            waiting_list = []
        waiting_list.append(id)
        item = self.items.get(id)
        item_config = item.get_config()
        ref_ids = self.find_refs_in_config(config=item_config, id=id)

        # if current item has reference already in the waiting list, that's circular references
        for d in ref_ids:
            if d in waiting_list:
                raise ValueError(f"detected circular references for id='{d}' in the config content.")

        if len(ref_ids) > 0:
            # # check whether the component has any unresolved deps
            for ref_id in ref_ids:
                if ref_id not in self.resolved_content:
                    # this reffring component is not resolved
                    if ref_id not in self.items:
                        raise RuntimeError(f"the referring item `{ref_id}` is not defined in config.")
                    # resolve the reference first
                    self.resolve_one_item(id=ref_id, waiting_list=waiting_list)

        # all references are resolved
        new_config = self.resolve_config_with_refs(config=item_config, id=id, refs=self.resolved_content)
        item.update_config(config=new_config)
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
        for k in self.items.keys():
            self.resolve_one_item(id=k)

    def get_resolved_content(self, id: str):
        """
        Get the resolved content with specified id name.
        If not resolved, try to resolve it first.

        Args:
            id: id name of the expected item.

        """
        if id not in self.resolved_content:
            self.resolve_one_item(id=id)
        return self.resolved_content.get(id)

    def get_item(self, id: str, resolve: bool = False):
        """
        Get the config item with specified id name, then can be used for lazy instantiation.
        If `resolve=True`, try to resolve it first.

        Args:
            id: id name of the expected config item.

        """
        if resolve and id not in self.resolved_content:
            self.resolve_one_item(id=id)
        return self.items.get(id)

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
    def resolve_refs_pattern(value: str, refs: Dict) -> str:
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
            if ConfigExpression.is_expression(value):
                # replace with local code and execute later
                value = value.replace(item, f"refs['{ref_id}']")
            elif value == item:
                if ref_id not in refs:
                    raise KeyError(f"can not find expected ID '{ref_id}' in the references.")
                value = refs[ref_id]
        return value

    @staticmethod
    def find_refs_in_config(
        config: Union[Dict, List, str],
        id: Optional[str] = None,
        refs: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Recursively search all the content of input config item to get the ids of references.
        References mean (1) referring to the ID of other item, can be extracted by `match_fn`, for example:
        `{"lr": "$@epoch / 100"}` with "@" mark, the referring IDs: `["epoch"]`. (2) if sub-item in the config
        is instantiable, treat it as reference because must instantiate it before resolving current config.
        For `dict` and `list`, recursively check the sub-items.

        Args:
            config: input config content to search.
            id: ID name for the input config, default to `None`.
            refs: list of the ID name of existing references, default to `None`.

        """
        refs_: List[str] = [] if refs is None else refs
        if isinstance(config, str):
            refs_ += ReferenceResolver.match_refs_pattern(value=config)

        if isinstance(config, list):
            for i, v in enumerate(config):
                sub_id = f"{id}#{i}" if id is not None else f"{i}"
                if ConfigComponent.is_instantiable(v) or ConfigExpression.is_expression(v):
                    refs_.append(sub_id)
                refs_ = ReferenceResolver.find_refs_in_config(v, sub_id, refs_)
        if isinstance(config, dict):
            for k, v in config.items():
                sub_id = f"{id}#{k}" if id is not None else f"{k}"
                if ConfigComponent.is_instantiable(v) or ConfigExpression.is_expression(v):
                    refs_.append(sub_id)
                refs_ = ReferenceResolver.find_refs_in_config(v, sub_id, refs_)
        return refs_

    @staticmethod
    def resolve_config_with_refs(
        config: Union[Dict, List, str],
        id: Optional[str] = None,
        refs: Optional[Dict] = None,
    ):
        """
        With all the references in `refs`, resolve the config content with them and return new config.

        Args:
            config: input config content to resolve.
            id: ID name for the input config, default to `None`.
            refs: all the referring components with ids, default to `None`.

        """
        refs_: Dict = {} if refs is None else refs
        if isinstance(config, str):
            config = ReferenceResolver.resolve_refs_pattern(config, refs)
        if isinstance(config, list):
            # all the items in the list should be replaced with the references
            ret_list: List = []
            for i, v in enumerate(config):
                sub_id = f"{id}#{i}" if id is not None else f"{i}"
                if ConfigComponent.is_instantiable(v) or ConfigExpression.is_expression(v):
                    ret_list.append(refs_[sub_id])
                else:
                    ret_list.append(ReferenceResolver.resolve_config_with_refs(v, sub_id, refs_))
            return ret_list
        if isinstance(config, dict):
            # all the items in the dict should be replaced with the references
            ret_dict: Dict = {}
            for k, v in config.items():
                sub_id = f"{id}#{k}" if id is not None else f"{k}"
                if ConfigComponent.is_instantiable(v) or ConfigExpression.is_expression(v):
                    ret_dict[k] = refs_[sub_id]
                else:
                    ret_dict[k] = ReferenceResolver.resolve_config_with_refs(v, sub_id, refs_)
            return ret_dict
        return config

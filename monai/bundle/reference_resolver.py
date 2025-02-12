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

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator, Sequence
from typing import Any

from monai.bundle.config_item import ConfigComponent, ConfigExpression, ConfigItem
from monai.bundle.utils import DEPRECATED_ID_MAPPING, ID_REF_KEY, ID_SEP_KEY
from monai.utils import allow_missing_reference, look_up_option

__all__ = ["ReferenceResolver"]


class ReferenceResolver:
    """
    Utility class to manage a set of ``ConfigItem`` and resolve the references between them.

    This class maintains a set of ``ConfigItem`` objects and their associated IDs.
    The IDs must be unique within this set. A string in ``ConfigItem``
    starting with ``@`` will be treated as a reference to other ``ConfigItem`` objects by ID.
    Since ``ConfigItem`` may have a nested dictionary or list structure,
    the reference string may also contain the separator ``::`` to refer to a substructure by
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

    _vars = "__local_refs"
    sep = ID_SEP_KEY  # separator for key indexing
    ref = ID_REF_KEY  # reference prefix
    # match a reference string, e.g. "@id::key", "@id::key::0", "@_target_::key"
    id_matcher = re.compile(rf"{ref}(?:\w*)(?:{sep}\w*)*")
    # if `allow_missing_reference` and can't find a reference ID, will just raise a warning and don't update the config
    allow_missing_reference = allow_missing_reference

    def __init__(self, items: Sequence[ConfigItem] | None = None):
        # save the items in a dictionary with the `ConfigItem.id` as key
        self.items: dict[str, ConfigItem] = {} if items is None else {i.get_id(): i for i in items}
        self.resolved_content: dict[str, ConfigExpression | str | Any | None] = {}

    def reset(self):
        """
        Clear all the added `ConfigItem` and all the resolved content.

        """
        self.items = {}
        self.resolved_content = {}

    def is_resolved(self) -> bool:
        return bool(self.resolved_content)

    def add_item(self, item: ConfigItem) -> None:
        """
        Add a ``ConfigItem`` to the resolver.

        Args:
            item: a ``ConfigItem``.

        """
        id = item.get_id()
        if id in self.items:
            return
        self.items[id] = item

    def get_item(self, id: str, resolve: bool = False, **kwargs: Any) -> ConfigItem | None:
        """
        Get the ``ConfigItem`` by id.

        If ``resolve=True``, the returned item will be resolved, that is,
        all the reference strings are substituted by the corresponding ``ConfigItem`` objects.

        Args:
            id: id of the expected config item.
            resolve: whether to resolve the item if it is not resolved, default to False.
            kwargs: keyword arguments to pass to ``_resolve_one_item()``.
                Currently support ``instantiate`` and ``eval_expr``. Both are defaulting to True.
        """
        id = self.normalize_id(id)
        if resolve and id not in self.resolved_content:
            self._resolve_one_item(id=id, **kwargs)
        return self.items.get(id)

    def _resolve_one_item(
        self, id: str, waiting_list: set[str] | None = None, **kwargs: Any
    ) -> ConfigExpression | str | Any | None:
        """
        Resolve and return one ``ConfigItem`` of ``id``, cache the resolved result in ``resolved_content``.
        If it has unresolved references, recursively resolve the referring items first.

        Args:
            id: id name of ``ConfigItem`` to be resolved.
            waiting_list: set of ids pending to be resolved.
                It's used to detect circular references such as:
                `{"name": "A", "dep": "@B"}` and `{"name": "B", "dep": "@A"}`.
            kwargs: keyword arguments to pass to ``_resolve_one_item()``.
                Currently support ``instantiate``, ``eval_expr`` and ``default``.
                `instantiate` and `eval_expr` are defaulting to True, `default` is the target config item
                if the `id` is not in the config content, must be a `ConfigItem` object.

        """
        id = self.normalize_id(id)
        if id in self.resolved_content:
            return self.resolved_content[id]
        try:
            item = look_up_option(id, self.items, print_all_options=False, default=kwargs.get("default", "no_default"))
        except ValueError as err:
            raise KeyError(f"id='{id}' is not found in the config resolver.") from err
        if not isinstance(item, ConfigItem):
            return item
        item_config = item.get_config()

        if waiting_list is None:
            waiting_list = set()
        waiting_list.add(id)

        for t, v in self.items.items():
            if (
                t not in self.resolved_content
                and isinstance(v, ConfigExpression)
                and v.is_import_statement(v.get_config())
            ):
                self.resolved_content[t] = v.evaluate() if kwargs.get("eval_expr", True) else v
        for d in self.find_refs_in_config(config=item_config, id=id).keys():
            # if current item has reference already in the waiting list, that's circular references
            if d in waiting_list:
                raise ValueError(f"detected circular references '{d}' for id='{id}' in the config content.")
            # check whether the component has any unresolved references
            if d not in self.resolved_content:
                # this referring item is not resolved
                try:
                    look_up_option(d, self.items, print_all_options=False)
                except ValueError as err:
                    msg = f"the referring item `@{d}` is not defined in the config content."
                    if not self.allow_missing_reference:
                        raise ValueError(msg) from err
                    warnings.warn(msg)
                    continue
                # recursively resolve the reference first
                self._resolve_one_item(id=d, waiting_list=waiting_list, **kwargs)
                waiting_list.discard(d)

        # all references are resolved, then try to resolve current config item
        new_config = self.update_config_with_refs(config=item_config, id=id, refs=self.resolved_content)
        item.update_config(config=new_config)
        # save the resolved result into `resolved_content` to recursively resolve others
        if isinstance(item, ConfigComponent):
            self.resolved_content[id] = item.instantiate() if kwargs.get("instantiate", True) else item
        elif isinstance(item, ConfigExpression):
            run_eval = kwargs.get("eval_expr", True)
            self.resolved_content[id] = (
                item.evaluate(globals={f"{self._vars}": self.resolved_content}) if run_eval else item
            )
        else:
            self.resolved_content[id] = new_config
        return self.resolved_content[id]

    def get_resolved_content(self, id: str, **kwargs: Any) -> ConfigExpression | str | Any | None:
        """
        Get the resolved ``ConfigItem`` by id.

        Args:
            id: id name of the expected item.
            kwargs: keyword arguments to pass to ``_resolve_one_item()``.
                Currently support ``instantiate``, ``eval_expr`` and ``default``.
                `instantiate` and `eval_expr` are defaulting to True, `default` is the target config item
                if the `id` is not in the config content, must be a `ConfigItem` object.

        """
        return self._resolve_one_item(id=id, **kwargs)

    def remove_resolved_content(self, id: str) -> Any | None:
        """
        Remove the resolved ``ConfigItem`` by id.

        Args:
            id: id name of the expected item.

        """
        return self.resolved_content.pop(id) if id in self.resolved_content else None

    @classmethod
    def normalize_id(cls, id: str | int) -> str:
        """
        Normalize the id string to consistently use `cls.sep`.

        Args:
            id: id string to be normalized.
        """
        return str(id).replace("#", cls.sep)  # backward compatibility `#` is the old separator

    def normalize_meta_id(self, config: Any) -> Any:
        """
        Update deprecated identifiers in `config` using `DEPRECATED_ID_MAPPING`.
        This will replace names that are marked as deprecated with their replacement.

        Args:
            config: input config to be updated.
        """
        if isinstance(config, dict):
            for _id, _new_id in DEPRECATED_ID_MAPPING.items():
                if _id in config.keys():
                    warnings.warn(
                        f"Detected deprecated name '{_id}' in configuration file, replacing with '{_new_id}'."
                    )
                    config[_new_id] = config.pop(_id)
        return config

    @classmethod
    def split_id(cls, id: str | int, last: bool = False) -> list[str]:
        """
        Split the id string into a list of strings by `cls.sep`.

        Args:
            id: id string to be split.
            last: whether to split the rightmost part of the id. default is False (split all parts).
        """
        if not last:
            return cls.normalize_id(id).split(cls.sep)
        res = cls.normalize_id(id).rsplit(cls.sep, 1)
        return ["".join(res[:-1]), res[-1]]

    @classmethod
    def iter_subconfigs(cls, id: str, config: Any) -> Iterator[tuple[str, str, Any]]:
        """
        Iterate over the sub-configs of the input config, the output `sub_id` uses `cls.sep` to denote substructure.

        Args:
            id: id string of the current input config.
            config: input config to be iterated.
        """
        for k, v in config.items() if isinstance(config, dict) else enumerate(config):
            sub_id = f"{id}{cls.sep}{k}" if id != "" else f"{k}"
            yield k, sub_id, v

    @classmethod
    def match_refs_pattern(cls, value: str) -> dict[str, int]:
        """
        Match regular expression for the input string to find the references.
        The reference string starts with ``"@"``, like: ``"@XXX::YYY::ZZZ"``.

        Args:
            value: input value to match regular expression.

        """
        refs: dict[str, int] = {}
        # regular expression pattern to match "@XXX" or "@XXX::YYY"
        value = cls.normalize_id(value)
        result = cls.id_matcher.findall(value)
        value_is_expr = ConfigExpression.is_expression(value)
        for item in result:
            if value_is_expr or value == item:
                # only check when string starts with "$" or the whole content is "@XXX"
                id = item[len(cls.ref) :]
                refs[id] = refs.get(id, 0) + 1
        return refs

    @classmethod
    def update_refs_pattern(cls, value: str, refs: dict) -> str:
        """
        Match regular expression for the input string to update content with the references.
        The reference part starts with ``"@"``, like: ``"@XXX::YYY::ZZZ"``.
        References dictionary must contain the referring IDs as keys.

        Args:
            value: input value to match regular expression.
            refs: all the referring components with ids as keys, default to `None`.

        """
        # regular expression pattern to match "@XXX" or "@XXX::YYY"
        value = cls.normalize_id(value)
        result = cls.id_matcher.findall(value)
        # reversely sort the matched references by length
        # and handle the longer first in case a reference item is substring of another longer item
        result.sort(key=len, reverse=True)
        value_is_expr = ConfigExpression.is_expression(value)
        for item in result:
            # only update reference when string starts with "$" or the whole content is "@XXX"
            if value_is_expr or value == item:
                ref_id = item[len(cls.ref) :]  # remove the ref prefix "@"
                if ref_id not in refs:
                    msg = f"can not find expected ID '{ref_id}' in the references."
                    if not cls.allow_missing_reference:
                        raise KeyError(msg)
                    warnings.warn(msg)
                    continue
                if value_is_expr:
                    # replace with local code, `{"__local_refs": self.resolved_content}` will be added to
                    # the `globals` argument of python `eval` in the `evaluate`
                    value = value.replace(item, f"{cls._vars}['{ref_id}']")
                elif value == item:
                    # the whole content is "@XXX", it will avoid the case that regular string contains "@"
                    value = refs[ref_id]
        return value

    @classmethod
    def find_refs_in_config(cls, config: Any, id: str, refs: dict[str, int] | None = None) -> dict[str, int]:
        """
        Recursively search all the content of input config item to get the ids of references.
        References mean: the IDs of other config items (``"@XXX"`` in this config item), or the
        sub-item in the config is `instantiable`, or the sub-item in the config is `expression`.
        For `dict` and `list`, recursively check the sub-items.

        Args:
            config: input config content to search.
            id: ID name for the input config item.
            refs: dict of the ID name and count of found references, default to `None`.

        """
        refs_: dict[str, int] = refs or {}
        if isinstance(config, str):
            for id, count in cls.match_refs_pattern(value=config).items():  # ref count is not currently used
                refs_[id] = refs_.get(id, 0) + count
        if not isinstance(config, (list, dict)):
            return refs_
        for _, sub_id, v in cls.iter_subconfigs(id, config):
            if ConfigComponent.is_instantiable(v) or ConfigExpression.is_expression(v) and sub_id not in refs_:
                refs_[sub_id] = 1
            refs_ = cls.find_refs_in_config(v, sub_id, refs_)
        return refs_

    @classmethod
    def update_config_with_refs(cls, config: Any, id: str, refs: dict | None = None) -> Any:
        """
        With all the references in ``refs``, update the input config content with references
        and return the new config.

        Args:
            config: input config content to update.
            id: ID name for the input config.
            refs: all the referring content with ids, default to `None`.

        """
        refs_: dict = refs or {}
        if isinstance(config, str):
            return cls.update_refs_pattern(config, refs_)
        if not isinstance(config, (list, dict)):
            return config
        ret = type(config)()
        for idx, sub_id, v in cls.iter_subconfigs(id, config):
            if ConfigComponent.is_instantiable(v) or ConfigExpression.is_expression(v):
                updated = refs_[sub_id]
                if ConfigComponent.is_instantiable(v) and updated is None:
                    # the component is disabled
                    continue
            else:
                updated = cls.update_config_with_refs(v, sub_id, refs_)
            ret.update({idx: updated}) if isinstance(ret, dict) else ret.append(updated)
        return ret

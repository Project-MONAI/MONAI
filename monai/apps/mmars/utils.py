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
from typing import Callable, Dict, List, Optional, Union


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
        if is_expression(value) or value == item:
            # only check when string starts with "$" or the whole content is "@XXX"
            ref_obj_id = item[1:]
            if ref_obj_id not in refs:
                refs.append(ref_obj_id)
    return refs


def resolve_refs_pattern(value: str, refs: Dict, globals: Optional[Dict] = None) -> str:
    """
    Match regular expression for the input string to update content with the references.
    The reference part starts with "@", like: "@XXX#YYY#ZZZ".
    References dictionary must contain the referring IDs as keys.

    Args:
        value: input value to match regular expression.
        refs: all the referring components with ids as keys, default to `None`.
        globals: predefined global variables to execute code string with `eval()`.

    """
    # regular expression pattern to match "@XXX" or "@XXX#YYY"
    result = re.compile(r"@\w*[\#\w]*").findall(value)
    for item in result:
        ref_id = item[1:]
        if is_expression(value):
            # replace with local code and execute later
            value = value.replace(item, f"refs['{ref_id}']")
        elif value == item:
            if ref_id not in refs:
                raise KeyError(f"can not find expected ID '{ref_id}' in the references.")
            value = refs[ref_id]
    if is_expression(value):
        # execute the code string with python `eval()`
        value = eval(value[1:], globals, {"refs": refs})
    return value


def find_refs_in_config(
    config: Union[Dict, List, str],
    id: Optional[str] = None,
    refs: Optional[List[str]] = None,
    match_fn: Callable = match_refs_pattern,
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
        match_fn: callable function to match config item for references, take `config` as parameter.

    """
    refs_: List[str] = [] if refs is None else refs
    if isinstance(config, str):
        refs_ += match_fn(value=config)

    if isinstance(config, list):
        for i, v in enumerate(config):
            sub_id = f"{id}#{i}" if id is not None else f"{i}"
            if is_instantiable(v):
                refs_.append(sub_id)
            refs_ = find_refs_in_config(v, sub_id, refs_, match_fn)
    if isinstance(config, dict):
        for k, v in config.items():
            sub_id = f"{id}#{k}" if id is not None else f"{k}"
            if is_instantiable(v):
                refs_.append(sub_id)
            refs_ = find_refs_in_config(v, sub_id, refs_, match_fn)
    return refs_


def resolve_config_with_refs(
    config: Union[Dict, List, str],
    id: Optional[str] = None,
    refs: Optional[Dict] = None,
    globals: Optional[Dict] = None,
    match_fn: Callable = resolve_refs_pattern,
):
    """
    With all the references in `refs`, resolve the config content with them and return new config.

    Args:
        config: input config content to resolve.
        id: ID name for the input config, default to `None`.
        refs: all the referring components with ids, default to `None`.
        globals: predefined global variables to execute code string with `eval()`.
        match_fn: callable function to match config item for references, take `config`,
            `refs` and `globals` as parameters.

    """
    refs_: Dict = {} if refs is None else refs
    if isinstance(config, str):
        config = match_fn(config, refs, globals)
    if isinstance(config, list):
        # all the items in the list should be replaced with the references
        ret_list: List = []
        for i, v in enumerate(config):
            sub = f"{id}#{i}" if id is not None else f"{i}"
            ret_list.append(
                refs_[sub] if is_instantiable(v) else resolve_config_with_refs(v, sub, refs_, globals, match_fn)
            )
        return ret_list
    if isinstance(config, dict):
        # all the items in the dict should be replaced with the references
        ret_dict: Dict = {}
        for k, v in config.items():
            sub = f"{id}#{k}" if id is not None else f"{k}"
            ret_dict.update(
                {k: refs_[sub] if is_instantiable(v) else resolve_config_with_refs(v, sub, refs_, globals, match_fn)}
            )
        return ret_dict
    return config


def is_instantiable(config: Union[Dict, List, str]) -> bool:
    """
    Check whether the content of the config represents a `class` or `function` to instantiate
    with specified "<path>" or "<name>".

    Args:
        config: input config content to check.

    """
    return isinstance(config, dict) and ("<path>" in config or "<name>" in config)


def is_expression(config: Union[Dict, List, str]) -> bool:
    """
    Check whether the content of the config is executable expression string.
    If True, the string should start with "$" mark.

    Args:
        config: input config content to check.

    """
    return isinstance(config, str) and config.startswith("$")

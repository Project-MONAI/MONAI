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


def is_to_build(config: Union[Dict, List, str]) -> bool:
    """
    Check whether the target component of the config is a `class` or `function` to build
    with specified "<path>" or "<name>".

    Args:
        config: input config content to check.

    """
    return isinstance(config, dict) and ("<path>" in config or "<name>" in config)


def search_config_with_deps(
    config: Union[Dict, List, str], id: Optional[str] = None, deps: Optional[List[str]] = None
) -> List[str]:
    """
    Recursively search all the content of input config compoent to get the ids of dependencies.
    It's used to build all the dependencies before build current config component.
    For `dict` and `list`, recursively check the sub-items.
    For example: `{"<name>": "DataLoader", "<args>": {"dataset": "@dataset"}}`, the dependency IDs: `["dataset"]`.

    Args:
        config: input config content to search.
        id: ID name for the input config, default to `None`.
        deps: list of the ID name of existing dependencies, default to `None`.

    """
    deps_: List[str] = [] if deps is None else deps
    pattern = re.compile(r"@\w*[\#\w]*")  # match ref as args: "@XXX#YYY#ZZZ"
    if isinstance(config, list):
        for i, v in enumerate(config):
            sub_id = f"{id}#{i}" if id is not None else f"{i}"
            if is_to_build(v):
                # sub-item is component need to build, mark as dependency
                deps_.append(sub_id)
            deps_ = search_config_with_deps(v, sub_id, deps_)
    if isinstance(config, dict):
        for k, v in config.items():
            sub_id = f"{id}#{k}" if id is not None else f"{k}"
            if is_to_build(v):
                # sub-item is component need to build, mark as dependency
                deps_.append(sub_id)
            deps_ = search_config_with_deps(v, sub_id, deps_)
    if isinstance(config, str):
        result = pattern.findall(config)
        for item in result:
            if config.startswith("$") or config == item:
                # only check when string starts with "$" or the whole content is "@XXX"
                ref_obj_id = item[1:]
                if ref_obj_id not in deps_:
                    deps_.append(ref_obj_id)
    return deps_


def resolve_config_with_deps(
    config: Union[Dict, List, str],
    deps: Optional[Dict] = None,
    id: Optional[str] = None,
    globals: Optional[Dict] = None,
):
    """
    With all the dependencies in `deps`, resolve the config content with them and return new config.

    Args:
        config: input config content to resolve.
        deps: all the dependent components with ids, default to `None`.
        id: id name for the input config, default to `None`.
        globals: predefined global variables to execute code string with `eval()`.

    """
    deps_: Dict = {} if deps is None else deps
    pattern = re.compile(r"@\w*[\#\w]*")  # match ref as args: "@XXX#YYY#ZZZ"
    if isinstance(config, list):
        # all the items in the list should be replaced with the reference
        ret_list: List = []
        for i, v in enumerate(config):
            sub_id = f"{id}#{i}" if id is not None else f"{i}"
            ret_list.append(deps_[sub_id] if is_to_build(v) else resolve_config_with_deps(v, deps_, sub_id, globals))
        return ret_list
    if isinstance(config, dict):
        # all the items in the dict should be replaced with the reference
        ret_dict: Dict = {}
        for k, v in config.items():
            sub_id = f"{id}#{k}" if id is not None else f"{k}"
            ret_dict[k] = deps_[sub_id] if is_to_build(v) else resolve_config_with_deps(v, deps_, sub_id, globals)
        return ret_dict
    if isinstance(config, str):
        result = pattern.findall(config)
        config_: str = config  # to avoid mypy CI errors
        for item in result:
            ref_obj_id = item[1:]
            if config_.startswith("$"):
                # replace with local code and execute later
                config_ = config_.replace(item, f"deps_['{ref_obj_id}']")
            elif config_ == item:
                config_ = deps_[ref_obj_id]
        if isinstance(config_, str) and config_.startswith("$"):
            config_ = eval(config_[1:], globals, {"deps_": deps_})
        return config_
    return config

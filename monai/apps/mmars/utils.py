# Copyright 2020 - 2021 MONAI Consortium
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


def search_configs_with_deps(config: Union[Dict, List, str], id: str, deps: Optional[List[str]] = None) -> List[str]:
    """
    Recursively search all the content of input config compoent to get the ids of dependencies.
    It's used to build all the dependencies before build current config component.
    For `dict` and `list`, treat every item as a dependency.
    For example, for `{"<name>": "DataLoader", "<args>": {"dataset": "@dataset"}}`, the dependency ids:
    `["<name>", "<args>", "<args>#dataset", "dataset"]`.

    Args:
        config: input config content to search.
        id: id name for the input config.
        deps: list of the id name of existing dependencies, default to None.

    """
    deps_: List[str] = [] if deps is None else deps
    pattern = re.compile(r"@\w*[\#\w]*")  # match ref as args: "@XXX#YYY#ZZZ"
    if isinstance(config, list):
        for i, v in enumerate(config):
            sub_id = f"{id}#{i}"
            # all the items in the list should be marked as dependent reference
            deps_.append(sub_id)
            deps_ = search_configs_with_deps(v, sub_id, deps_)
    if isinstance(config, dict):
        for k, v in config.items():
            sub_id = f"{id}#{k}"
            # all the items in the dict should be marked as dependent reference
            deps_.append(sub_id)
            deps_ = search_configs_with_deps(v, sub_id, deps_)
    if isinstance(config, str):
        result = pattern.findall(config)
        for item in result:
            if config.startswith("$") or config == item:
                ref_obj_id = item[1:]
                if ref_obj_id not in deps_:
                    deps_.append(ref_obj_id)
    return deps_


def update_configs_with_deps(config: Union[Dict, List, str], deps: dict, id: str, globals: Optional[Dict] = None):
    """
    With all the dependencies in `deps`, update the config content with them and return new config.
    It can be used for lazy instantiation.

    Args:
        config: input config content to update.
        deps: all the dependent components with ids.
        id: id name for the input config.
        globals: predefined global variables to execute code string with `eval()`.

    """
    pattern = re.compile(r"@\w*[\#\w]*")  # match ref as args: "@XXX#YYY#ZZZ"
    if isinstance(config, list):
        # all the items in the list should be replaced with the reference
        return [deps[f"{id}#{i}"] for i in range(len(config))]
    if isinstance(config, dict):
        # all the items in the dict should be replaced with the reference
        return {k: deps[f"{id}#{k}"] for k, _ in config.items()}
    if isinstance(config, str):
        result = pattern.findall(config)
        config_: str = config
        for item in result:
            ref_obj_id = item[1:]
            if config_.startswith("$"):
                # replace with local code and execute soon
                config_ = config_.replace(item, f"deps['{ref_obj_id}']")
            elif config_ == item:
                config_ = deps[ref_obj_id]

        if isinstance(config_, str):
            if config_.startswith("$"):
                config_ = eval(config_[1:], globals, {"deps": deps})
        return config_
    return config

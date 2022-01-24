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

import importlib
from optparse import Option
import re
from typing import Dict, List, Optional, Union


def get_class(class_path: str):
    """
    Get the class from specified class path.

    Args:
        class_path (str): full path of the class.

    Raises:
        ValueError: invalid class_path, missing the module name.
        ValueError: class does not exist.
        ValueError: module does not exist.

    """
    if len(class_path.split(".")) < 2:
        raise ValueError(f"invalid class_path: {class_path}, missing the module name.")
    module_name, class_name = class_path.rsplit(".", 1)

    try:
        module_ = importlib.import_module(module_name)

        try:
            class_ = getattr(module_, class_name)
        except AttributeError as e:
            raise ValueError(f"class {class_name} does not exist.") from e

    except AttributeError as e:
        raise ValueError(f"module {module_name} does not exist.") from e

    return class_


def instantiate_class(class_path: str, **kwargs):
    """
    Method for creating an instance for the specified class.

    Args:
        class_path: full path of the class.
        kwargs: arguments to initialize the class instance.

    Raises:
        ValueError: class has paramenters error.
    """

    try:
        return get_class(class_path)(**kwargs)
    except TypeError as e:
        raise ValueError(f"class {class_path} has parameters error.") from e


def search_configs_with_objs(config: Union[Dict, List, str], refs: List[str], id: str):
    pattern = re.compile(r'@\w*[\#\w]*')  # match ref as args: "@XXX#YYY#ZZZ"
    if isinstance(config, list):
        for i, v in enumerate(config):
            sub_id = f"{id}#{i}"
            # all the items in the list should be marked as dependent reference
            refs.append(sub_id)
            refs = search_configs_with_objs(v, refs, sub_id)
    if isinstance(config, dict):
        for k, v in config.items():
            sub_id = f"{id}#{k}"
            # all the items in the dict should be marked as dependent reference
            refs.append(sub_id)
            refs = search_configs_with_objs(v, refs, sub_id)
    if isinstance(config, str):
        result = pattern.findall(config)
        for item in result:
            if config.startswith("$") or config == item:
                ref_obj_id = item[1:]
                if ref_obj_id not in refs:
                    refs.append(ref_obj_id)
    return refs


def update_configs_with_objs(config: Union[Dict, List, str], refs: dict, id: str, globals: Optional[Dict] = None):
    pattern = re.compile(r'@\w*[\#\w]*')  # match ref as args: "@XXX#YYY#ZZZ"
    if isinstance(config, list):
        # all the items in the list should be replaced with the reference
        config = [refs[f"{id}#{i}"] for i in range(len(config))]
    if isinstance(config, dict):
        # all the items in the dict should be replaced with the reference
        config = {k: refs[f"{id}#{k}"] for k, _ in config.items()}
    if isinstance(config, str):
        result = pattern.findall(config)
        for item in result:
            ref_obj_id = item[1:]
            if config.startswith("$"):
                # replace with local code and execute soon
                config = config.replace(item, f"refs['{ref_obj_id}']")
            elif config == item:
                config = refs[ref_obj_id]

        if isinstance(config, str):
            if config.startswith("$"):
                config = eval(config[1:], globals, {"refs": refs})
    return config

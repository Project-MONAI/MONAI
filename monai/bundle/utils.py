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

import json
import re
from distutils.util import strtobool
from typing import Any, Dict, Tuple, Union

from monai.bundle.config_parser import ConfigParser
from monai.utils import optional_import

yaml, _ = optional_import("yaml")


def load_config_file(filepath: str, **kwargs):
    """
    Load structured config file with the specified file path.
    Suppprt JSON and YAML formats.

    Args:
        filepath: path of target file to load, supported postfixes: `.json`, `.yml`, `.yaml`.
        kwargs: other arguments for `json.load` or `yaml.load`, depends on file format.
            for more details, please check:
            https://docs.python.org/3/library/json.html#json.load.
            https://pyyaml.org/wiki/PyYAMLDocumentation.

    """
    with open(filepath) as f:
        if filepath.lower().endswith(".json"):
            return json.load(f, **kwargs)
        if filepath.lower().endswith((".yml", ".yaml")):
            return yaml.safe_load(f, **kwargs)
        raise ValueError("only support JSON or YAML config file so far.")


def load_config_file_content(path: str, **kwargs):
    """
    Load part of the content from a config file with specified `id` in the path.
    If no `id` provided, load the whole content of the file.
    Suppprt JSON and YAML formats file.

    Args:
        path: path of target file to load, it can only load part of it appending target `id`
            in the path with "#" mark. for example: `/data/config.json`, `/data/config.json#net#<args>`.
        kwargs: other arguments for `json.load` or `yaml.load`, depends on file format.
            for more details, please check:
            https://docs.python.org/3/library/json.html#json.load.
            https://pyyaml.org/wiki/PyYAMLDocumentation.

    """
    pattern = r"(json|yaml|yml)"
    result = re.findall(pattern, path, re.IGNORECASE)
    if len(result) != 1:
        raise ValueError(f"path should only contain 1 file, but got: {path}.")

    # split the path into filepath and target id of the content
    paths = path.split(result[0])
    parser = ConfigParser(config=load_config_file(paths[0] + result[0], **kwargs))
    return parser[paths[1][1:] if paths[1] != "" else ""]


def parse_id_value(pair: str) -> Tuple[str, Any]:
    """
    Parse the "id:value" pair string to `id` and `value`.
    Will try to convert the correct data type of `value` from string.

    Args:
        pair: input "id:value" pair to parse.

    """
    items = pair.split(":")
    # remove blanks around id
    id = items[0].strip()
    value: Union[str, int, float, bool] = ""
    if len(items) > 1:
        # rejoin the rest, and remove blanks around value
        value = ":".join(items[1:]).strip()

    # try to convert the correct data type
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            try:
                value = bool(strtobool(str(value)))
            except ValueError:
                pass
    return id, value


def id_value_str_to_dict(pairs: str) -> Dict[str, Any]:
    """
    Utility to convert a string which represents a dict of `id:value` pairs to a python dict. For example:
    `"{postprocessing#<args>#postfix: output, network: <file>other.json#net_args}"`
    Will try to convert the data type of `value` from string to real type.

    """
    return dict(map(parse_id_value, pairs[1: -1].split(",")))

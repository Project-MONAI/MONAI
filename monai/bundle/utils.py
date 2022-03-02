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
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from monai.bundle.config_parser import ConfigParser
from monai.utils import ensure_tuple, optional_import

yaml, _ = optional_import("yaml")


def load_config_file(filepath: str, **kwargs):
    """
    Load config file with specified file path.
    Suppprt JSON and YAML formats.

    Args:
        filepath: path of target file to load, supported postfixes: `.json`, `yml`, `yaml`.
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


def update_default_args(args: Optional[Union[str, Dict]] = None, **kwargs) -> Dict:
    """
    Update the `args` with the input `kwargs`.
    For dict data, recursively update the content based on the keys.

    Args:
        args: source args to update.
        kwargs: destination args to update.

    """
    args_: Dict = args if isinstance(args, dict) is None else {}  # type: ignore
    if isinstance(args, str):
        args_ = load_config_file_content(args)

    # recursively update the default args with new args
    for k, v in kwargs.items():
        args_[k] = update_default_args(args_[k], **v) if isinstance(v, dict) and isinstance(args_.get(k), dict) else v
    return args_


def parse_id_value(pair: str) -> Tuple[str, Any]:
    """
    Parse the "id:value" pair string to `id` and `value`.
    Will try to convert the correct data type of `value` from string.

    Args:
        pair (str): input "id:value" pair to parse.

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
    Will try to convert the correct data type of `value` from string.

    """
    return dict(map(parse_id_value, pairs[1:-1].split(",")))


def parse_config_file(
    config_file: Union[str, Sequence[str]], meta_file: Union[str, Sequence[str]], override: Optional[Dict] = None
) -> ConfigParser:
    """
    Read the config file, metadata file and override with specified `id=value` pairs.
    Put metadata in the config content with key "<meta>".
    The `id` in `override` identifies target position to override with the `value`.
    If `value` starts with "<file>", it will automatically read the `file`
    and use the content as `value`.

    Args:
        config_file: filepath of the config file, the config content must be a dictionary,
            if providing a list of files, wil merge the content of them.
        meta_file: filepath of the metadata file, the config content must be a dictionary,
            if providing a list of files, wil merge the content of them.
        override: dict of `{id: value}` pairs to override or add the config content.

    """
    config: Dict = {"<meta>": {}}
    for f in ensure_tuple(config_file):
        content = load_config_file(f)
        if not isinstance(content, dict):
            raise ValueError("input config content must be a dictionary.")
        config.update(content)

    for f in ensure_tuple(meta_file):
        content = load_config_file(f)
        if not isinstance(content, dict):
            raise ValueError("meta data content must be a dictionary.")
        config["<meta>"].update(content)

    parser = ConfigParser(config=config)

    if override is not None:
        for id, v in override.items():
            if isinstance(v, str) and v.startswith("<file>"):
                v = load_config_file_content(v[6:])
            parser[id] = v

    return parser

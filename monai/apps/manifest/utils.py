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

from distutils.util import strtobool
import json
from typing import Any, Dict, List, Tuple
import yaml

from monai.apps.manifest.config_parser import ConfigParser


def read_config(filepath: str):
    """
    Read config file with specified file path.
    Suppprt JSON and YAML formats.

    """
    with open(filepath) as f:
        if filepath.lower().endswith(".json"):
            return json.load(f)
        if filepath.lower().endswith((".yml", ".yaml")):
            return yaml.load(f, Loader=yaml.FullLoader)
        raise ValueError("only support JSON or YAML config file so far.")


def parse_id_value(pair: str) -> Tuple[str, Any]:
    """
    Parse the "id=value" pair to `id` and `value`.
    Will try to convert the correct data type of `value` from string.

    Args:
        pair (str): input "id=value" pair to parse.

    """
    items = pair.split("=")
    # we remove blanks around id
    id = items[0].strip()
    value = ""
    if len(items) > 1:
        # rejoin the rest
        value = "=".join(items[1:])

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


def parse_config_files(config_file: str, meta_file: str, override: Dict = {}) -> ConfigParser:
    """
    Read the config file, metadata file and override with specified `id=value` pairs.
    Put metadata in the config content with key "<meta>".
    The `id` in `override` identifies target position to override with the `value`.
    If `value` starts with "<file>", it will automatically read the `file`
    and use the content as `value`.

    Args:
        config_file: filepath of the config file.
        meta_file: filepath of the metadata file.
        override: dict of `{id: value}` pairs to override or add the config content.

    """
    config = read_config(config_file)
    if not isinstance(config, dict):
        raise ValueError("input config file must be a dictionary.")

    config["<meta>"] = read_config(meta_file)

    parser = ConfigParser(config=config)

    if len(override) > 0:
        for id, v in override.items():
            if isinstance(v, str) and v.startswith("<file>"):
                v = read_config(v[6:])
            parser[id] = v

    return parser

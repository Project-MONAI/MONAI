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
from distutils.util import strtobool
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from monai.apps.manifest.config_parser import ConfigParser
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
            if "Loader" not in kwargs:
                kwargs["Loader"] = yaml.FullLoader
            return yaml.load(f, **kwargs)
        raise ValueError("only support JSON or YAML config file so far.")


def parse_id_value(pair: str) -> Tuple[str, Any]:
    """
    Parse the "id=value" pair string to `id` and `value`.
    Will try to convert the correct data type of `value` from string.

    Args:
        pair (str): input "id=value" pair to parse.

    """
    items = pair.split("=")
    # remove blanks around id
    id = items[0].strip()
    value: Union[str, int, float, bool] = ""
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


def parse_config_files(
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
                v = load_config_file(v[6:])
            parser[id] = v

    return parser

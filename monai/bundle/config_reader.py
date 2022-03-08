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
from pathlib import Path
import re
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from monai.config import PathLike
from monai.bundle.config_parser import ConfigParser
from monai.utils import ensure_tuple, optional_import

yaml, _ = optional_import("yaml")


class ConfigReader:
    suffixes = ["json", "yaml", "yml"]
    macro = "%"  # macro prefix
    meta_key = "<meta>"  # field key to save meta data

    def __init__(self):
        self.config: Dict = {self.meta_key: {}}

    @classmethod
    def load_config_file(cls, filepath: PathLike, **kwargs):
        """
        Load config file with specified file path.
        Suppprt JSON and YAML formats.

        Args:
            filepath: path of target file to load, supported postfixes: `.json`, `.yml`, `.yaml`.
            kwargs: other arguments for `json.load` or `yaml.load`, depends on file format.
                for more details, please check:
                https://docs.python.org/3/library/json.html#json.load.
                https://pyyaml.org/wiki/PyYAMLDocumentation.

        """
        _filepath: str = str(Path(filepath))
        with open(_filepath) as f:
            if _filepath.lower().endswith(cls.suffixes[0]):
                return json.load(f, **kwargs)
            if _filepath.lower().endswith(tuple(cls.suffixes[1:])):
                return yaml.safe_load(f, **kwargs)
            raise ValueError("only support JSON or YAML config file so far.")

    def read_meta(self, f: Union[PathLike, Sequence[PathLike], Dict], **kwargs):
        content = {}
        if isinstance(f, dict):
            # already loaded in dict
            content = f
        else:
            for i in ensure_tuple(f):
                content.update(self.load_config_file(i, **kwargs))
        self.config[self.meta_key] = content

    def read_config(self, f: Union[PathLike, Sequence[PathLike], Dict], **kwargs):
        content = {self.meta_key: self.config[self.meta_key]}
        if isinstance(f, dict):
            # already loaded in dict
            content.update(f)
        else:
            for i in ensure_tuple(f):
                content.update(self.load_config_file(i, **kwargs))
        self.config = content

    def get(self):
        return self.config

    def override(self, data: Dict[str, Any]):
        parser = ConfigParser(config=self.config)
        for id, v in data.items():
            parser[id] = v

    @classmethod
    def split_file_path_id(cls, path: str) -> Optional[Tuple[str, str]]:
        pattern = "|".join(cls.suffixes)
        result = re.findall(pattern, path, re.IGNORECASE)
        if len(result) != 1:
            # path should only contain 1 file
            return None
        paths = path.split(result[0])
        # return file path and target id
        return paths[0] + result[0], paths[1][1:] if paths[1] != "" else ""

    def _do_resolve(self, config, **kwargs):
        if isinstance(config, (dict, list)):
            subs = enumerate(config) if isinstance(config, list) else config.items()
            for k, v in subs:
                config[k] = self._do_resolve(v, **kwargs)
        if isinstance(config, str) and config.startswith(self.macro):
            # only support macro mark at the beginning of a string
            id = config[len(self.macro):]
            paths = self.split_file_path_id(id)
            if paths is None:
                # id is in the current config file
                parser = ConfigParser(config=self.config)
                data = deepcopy(parser[id])
            else:
                # id is in another config file
                parser = ConfigParser(config=self.load_config_file(paths[0], **kwargs))
                data = parser[paths[1]]
            # recursively check the resolved content
            return self._do_resolve(data, **kwargs)
        return config

    def resolve_macro(self, **kwargs):
        self.config = self._do_resolve(config=deepcopy(self.config))

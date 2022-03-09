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
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

from monai.config import PathLike
from monai.utils import ensure_tuple, optional_import

yaml, _ = optional_import("yaml")

__all__ = ["ConfigReader"]


class ConfigReader:
    """
    Read metadata, config from structured JSON or YAML files.
    Support to override the config content with specified `id` and value.
    Support to resolve the macro tokens in the config content.

    """

    suffixes = ["json", "yaml", "yml"]
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
            kwargs: other arguments for `json.load` or `yaml.safe_load`, depends on file format.
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

    @classmethod
    def export_config_file(cls, config: Dict, filepath: PathLike, **kwargs):
        """
        Export the config content to the specified file path.
        Suppprt JSON and YAML formats.

        Args:
            config: source config content to export.
            filepath: target file path to save, supported postfixes: `.json`, `.yml`, `.yaml`.
            kwargs: other arguments for `json.dump` or `yaml.safe_dump`, depends on file format.
                for more details, please check:
                https://docs.python.org/3/library/json.html#json.dump.
                https://pyyaml.org/wiki/PyYAMLDocumentation.

        """
        _filepath: str = str(Path(filepath))
        with open(_filepath, "w") as f:
            if _filepath.lower().endswith(cls.suffixes[0]):
                return json.dump(config, f, **kwargs)
            if _filepath.lower().endswith(tuple(cls.suffixes[1:])):
                return yaml.safe_dump(config, f, **kwargs)
            raise ValueError("only support JSON or YAML config file so far.")

    @classmethod
    def extract_file_path(cls, src: str) -> Optional[Tuple[str, str]]:
        """
        extract a config file path from the source string, return path and the rest string.
        return `None` if can't find any config file path.

        Args:
            src: source string to extract, it can be a config file path with / without additional information.
                for example: "/data/config.json", "/data/config.json#net#<args>".

        """
        pattern = "|".join(cls.suffixes)
        result = re.findall(pattern, src, re.IGNORECASE)
        if len(result) != 1:
            # src should only contain 1 file
            return None
        items = src.split(result[0])
        # return file path and the rest
        return items[0] + result[0], items[1]

    def read_meta(self, f: Union[PathLike, Sequence[PathLike], Dict], **kwargs):
        """
        Read the metadata from specified JSON or YAML file.
        Will put metadata in the config content with key "<meta>".

        Args:
            f: filepath of the meta data file, the content must be a dictionary,
                if providing a list of files, wil merge the content of them.
                if providing a dictionary directly, use it as meta data.
            kwargs: other arguments for `json.load` or `yaml.safe_load`, depends on file format.
                for more details, please check:
                https://docs.python.org/3/library/json.html#json.load.
                https://pyyaml.org/wiki/PyYAMLDocumentation.

        """
        content = {}
        if isinstance(f, dict):
            # already loaded in dict
            content = f
        else:
            for i in ensure_tuple(f):
                content.update(self.load_config_file(i, **kwargs))
        self.config[self.meta_key] = content

    def read_config(self, f: Union[PathLike, Sequence[PathLike], Dict], **kwargs):
        """
        Read the config from specified JSON or YAML file.
        Will store the config content in the `self.config` property.

        Args:
            f: filepath of the config file, the content must be a dictionary,
                if providing a list of files, wil merge the content of them.
                if providing a dictionary directly, use it as config.
            kwargs: other arguments for `json.load` or `yaml.safe_load`, depends on file format.
                for more details, please check:
                https://docs.python.org/3/library/json.html#json.load.
                https://pyyaml.org/wiki/PyYAMLDocumentation.

        """
        content = {self.meta_key: self.config[self.meta_key]}
        if isinstance(f, dict):
            # already loaded in dict
            content.update(f)
        else:
            for i in ensure_tuple(f):
                content.update(self.load_config_file(i, **kwargs))
        self.config = content

    def get(self):
        """
        Get the loaded config content.

        """
        return self.config

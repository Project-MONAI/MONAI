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
from typing import Dict, Sequence, Tuple, Union

from monai.config import PathLike
from monai.utils import ensure_tuple, look_up_option, optional_import

yaml, _ = optional_import("yaml")

__all__ = ["ConfigReader"]


class ConfigReader:
    """
    Read config and metadata from JSON or YAML files.
    Support to override the config content with specified `id` and value.
    Support to resolve the macro tokens in the config content.

    See also:

        - https://docs.python.org/3/library/json.html
        - https://pyyaml.org/wiki/PyYAMLDocumentation

    """

    suffixes = ("json", "yaml", "yml")
    suffix_match = rf"\.({'|'.join(suffixes)})"
    path_match = rf"(.*{suffix_match}$)"
    meta_key = "_meta_"  # field key to save metadata
    sep = "#"  # separator for file path and the id of content in the file

    def __init__(self):
        self.config: Dict = {self.meta_key: {}}

    @classmethod
    def load_config_file(cls, filepath: PathLike, **kwargs):
        """
        Load config file with specified file path (currently support JSON and YAML files).

        Args:
            filepath: path of target file to load, supported postfixes: `.json`, `.yml`, `.yaml`.
            kwargs: other arguments for ``json.load`` or ```yaml.safe_load``, depends on the file format.

        """
        _filepath: str = str(Path(filepath))
        if not re.compile(cls.path_match, re.IGNORECASE).findall(_filepath):
            raise ValueError(f'unknown file input: "{filepath}"')
        with open(_filepath) as f:
            if _filepath.lower().endswith(cls.suffixes[0]):
                return json.load(f, **kwargs)
            if _filepath.lower().endswith(cls.suffixes[1:]):
                return yaml.safe_load(f, **kwargs)
            raise ValueError(f"only support JSON or YAML config file so far, got name {_filepath}.")

    @classmethod
    def load_config_files(cls, files: Union[PathLike, Sequence[PathLike], dict], **kwargs) -> dict:
        """
        Load config files into a single config dict.

        Args:
            files: path of target files to load, supported postfixes: `.json`, `.yml`, `.yaml`.
            kwargs: other arguments for ``json.load`` or ```yaml.safe_load``, depends on the file format.
        """
        if isinstance(files, dict):  # already a config dict
            return files
        content = {}
        for i in ensure_tuple(files):
            content.update(cls.load_config_file(i, **kwargs))
        return content

    @classmethod
    def export_config_file(cls, config: Dict, filepath: PathLike, fmt="json", **kwargs):
        """
        Export the config content to the specified file path (currently support JSON and YAML files).

        Args:
            config: source config content to export.
            filepath: target file path to save.
            fmt: format of config content, currently support ``"json"`` and ``"yaml"``.
            kwargs: other arguments for ``json.dump`` or ``yaml.safe_dump``, depends on the file format.

        """
        _filepath: str = str(Path(filepath))
        writer = look_up_option(fmt.lower(), {"json", "yaml"})
        with open(_filepath, "w") as f:
            if writer == "json":
                return json.dump(config, f, **kwargs)
            if writer == "yaml":
                return yaml.safe_dump(config, f, **kwargs)
            raise ValueError(f"only support JSON or YAML config file so far, got {writer}.")

    @classmethod
    def split_path_id(cls, src: str) -> Tuple[str, str]:
        """
        Split `src` string into two parts: a config file path and component id.
        The file path should end with `(json|yaml|yml)`. The component id should be separated by `#` if it exists.
        If no path or no id, return "".

        Args:
            src: source string to split.

        """
        result = re.compile(rf"(.*{cls.suffix_match}(?=(?:{cls.sep}.*)|$))", re.IGNORECASE).findall(src)
        if not result:
            return "", src  # the src is a pure id
        path_name = result[0][0]  # at most one path_name
        _, ids = src.rsplit(path_name, 1)
        return path_name, ids[len(cls.sep) :] if ids.startswith(cls.sep) else ""

    def read_meta(self, f: Union[PathLike, Sequence[PathLike], Dict], **kwargs):
        """
        Read the metadata from specified JSON or YAML file.
        The metadata as a dictionary will be stored at ``self.config["_meta_"]``.

        Args:
            f: filepath of the metadata file, the content must be a dictionary,
                if providing a list of files, wil merge the content of them.
                if providing a dictionary directly, use it as metadata.
            kwargs: other arguments for ``json.load`` or ``yaml.safe_load``, depends on the file format.

        """
        self.config[self.meta_key] = self.load_config_files(f, **kwargs)

    def read_config(self, f: Union[PathLike, Sequence[PathLike], Dict], **kwargs):
        """
        Read the config from specified JSON or YAML file.
        The config content in the `self.config` dictionary.

        Args:
            f: filepath of the config file, the content must be a dictionary,
                if providing a list of files, wil merge the content of them.
                if providing a dictionary directly, use it as config.
            kwargs: other arguments for ``json.load`` or ``yaml.safe_load``, depends on the file format.

        """
        content = {self.meta_key: self.config.get(self.meta_key)}
        content.update(self.load_config_files(f, **kwargs))
        self.config = content

    def get(self):
        """
        Get the loaded config content.

        """
        return self.config

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

from monai.bundle.reference_resolver import ReferenceResolver
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

        - https://docs.python.org/3/library/json.html#json.load
        - https://pyyaml.org/wiki/PyYAMLDocumentation

    """

    suffixes = ("json", "yaml", "yml")
    path_match = re.compile(rf"(.*\.({'|'.join(suffixes)})$)", re.IGNORECASE)
    meta_key = "<meta>"  # field key to save metadata

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
        if not cls.path_match.findall(_filepath):
            raise ValueError(f'unknown file input: "{filepath}"')
        with open(_filepath) as f:
            if _filepath.lower().endswith(cls.suffixes[0]):
                return json.load(f, **kwargs)
            if _filepath.lower().endswith(cls.suffixes[1:]):
                return yaml.safe_load(f, **kwargs)
            raise ValueError(f"only support JSON or YAML config file so far, got name {_filepath}.")

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

        Args:
            src: source string to split.

        """
        path, *ids = f"{src}".rsplit(ReferenceResolver.sep, 1)
        path_name = cls.path_match.findall(path)
        if not path_name:
            return "", src  # the src is a pure id
        if len(path_name) < 1 and len(path_name[0]) < 1:
            raise ValueError(f"invalid config file path: {path}")
        ids_string: str = ids[0] if ids else ""
        return path_name[0][0], ids_string

    def read_meta(self, f: Union[PathLike, Sequence[PathLike], Dict], **kwargs):
        """
        Read the metadata from specified JSON or YAML file.
        The metadata as a dictionary will be stored at ``self.config["<meta>"]``.

        Args:
            f: filepath of the metadata file, the content must be a dictionary,
                if providing a list of files, wil merge the content of them.
                if providing a dictionary directly, use it as metadata.
            kwargs: other arguments for ``json.load`` or ``yaml.safe_load``, depends on the file format.

        """
        if isinstance(f, dict):
            # already loaded in dict
            content = f
        else:
            content = {}
            for i in ensure_tuple(f):
                content.update(self.load_config_file(i, **kwargs))
        self.config[self.meta_key] = content

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

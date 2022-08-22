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
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from monai.bundle.config_item import ComponentLocator, ConfigComponent, ConfigExpression, ConfigItem
from monai.bundle.reference_resolver import ReferenceResolver
from monai.bundle.utils import ID_REF_KEY, ID_SEP_KEY, MACRO_KEY
from monai.config import PathLike
from monai.utils import ensure_tuple, look_up_option, optional_import

yaml, _ = optional_import("yaml")

__all__ = ["ConfigParser"]

_default_globals = {"monai": "monai", "torch": "torch", "np": "numpy", "numpy": "numpy"}


class ConfigParser:
    """
    The primary configuration parser. It traverses a structured config (in the form of nested Python dict or list),
    creates ``ConfigItem``, and assign unique IDs according to the structures.

    This class provides convenient access to the set of ``ConfigItem`` of the config by ID.
    A typical workflow of config parsing is as follows:

        - Initialize ``ConfigParser`` with the ``config`` source.
        - Call ``get_parsed_content()`` to get expected component with `id`.

    .. code-block:: python

        from monai.bundle import ConfigParser

        config = {
            "my_dims": 2,
            "dims_1": "$@my_dims + 1",
            "my_xform": {"_target_": "LoadImage"},
            "my_net": {"_target_": "BasicUNet", "spatial_dims": "@dims_1", "in_channels": 1, "out_channels": 4},
            "trainer": {"_target_": "SupervisedTrainer", "network": "@my_net", "preprocessing": "@my_xform"}
        }
        # in the example $@my_dims + 1 is an expression, which adds 1 to the value of @my_dims
        parser = ConfigParser(config)

        # get/set configuration content, the set method should happen before calling parse()
        print(parser["my_net"]["in_channels"])  # original input channels 1
        parser["my_net"]["in_channels"] = 4  # change input channels to 4
        print(parser["my_net"]["in_channels"])

        # instantiate the network component
        parser.parse(True)
        net = parser.get_parsed_content("my_net", instantiate=True)
        print(net)

        # also support to get the configuration content of parsed `ConfigItem`
        trainer = parser.get_parsed_content("trainer", instantiate=False)
        print(trainer)

    Args:
        config: input config source to parse.
        excludes: when importing modules to instantiate components,
            excluding components from modules specified in ``excludes``.
        globals: pre-import packages as global variables to ``ConfigExpression``,
            so that expressions, for example, ``"$monai.data.list_data_collate"`` can use ``monai`` modules.
            The current supported globals and alias names are
            ``{"monai": "monai", "torch": "torch", "np": "numpy", "numpy": "numpy"}``.
            These are MONAI's minimal dependencies. Additional packages could be included with `globals={"itk": "itk"}`.

    See also:

        - :py:class:`monai.bundle.ConfigItem`
        - :py:class:`monai.bundle.scripts.run`

    """

    suffixes = ("json", "yaml", "yml")
    suffix_match = rf".*\.({'|'.join(suffixes)})"
    path_match = rf"({suffix_match}$)"
    # match relative id names, e.g. "@#data", "@##transform#1"
    relative_id_prefix = re.compile(rf"(?:{ID_REF_KEY}|{MACRO_KEY}){ID_SEP_KEY}+")
    meta_key = "_meta_"  # field key to save metadata

    def __init__(
        self,
        config: Any = None,
        excludes: Optional[Union[Sequence[str], str]] = None,
        globals: Optional[Dict[str, Any]] = None,
    ):
        self.config = None
        self.globals: Dict[str, Any] = {}
        _globals = _default_globals.copy()
        if isinstance(_globals, dict) and globals is not None:
            _globals.update(globals)
        if _globals is not None:
            for k, v in _globals.items():
                self.globals[k] = optional_import(v)[0] if isinstance(v, str) else v

        self.locator = ComponentLocator(excludes=excludes)
        self.ref_resolver = ReferenceResolver()
        if config is None:
            config = {self.meta_key: {}}
        self.set(config=config)

    def __repr__(self):
        return f"{self.config}"

    def __getitem__(self, id: Union[str, int]):
        """
        Get the config by id.

        Args:
            id: id of the ``ConfigItem``, ``"#"`` in id are interpreted as special characters to
                go one level further into the nested structures.
                Use digits indexing from "0" for list or other strings for dict.
                For example: ``"xform#5"``, ``"net#channels"``. ``""`` indicates the entire ``self.config``.

        """
        if id == "":
            return self.config
        config = self.config
        for k in str(id).split(ID_SEP_KEY):
            if not isinstance(config, (dict, list)):
                raise ValueError(f"config must be dict or list for key `{k}`, but got {type(config)}: {config}.")
            indexing = k if isinstance(config, dict) else int(k)
            config = config[indexing]
        return config

    def __setitem__(self, id: Union[str, int], config: Any):
        """
        Set config by ``id``.  Note that this method should be used before ``parse()`` or ``get_parsed_content()``
        to ensure the updates are included in the parsed content.

        Args:
            id: id of the ``ConfigItem``, ``"#"`` in id are interpreted as special characters to
                go one level further into the nested structures.
                Use digits indexing from "0" for list or other strings for dict.
                For example: ``"xform#5"``, ``"net#channels"``. ``""`` indicates the entire ``self.config``.
            config: config to set at location ``id``.

        """
        if id == "":
            self.config = config
            self.ref_resolver.reset()
            return
        keys = str(id).split(ID_SEP_KEY)
        # get the last parent level config item and replace it
        last_id = ID_SEP_KEY.join(keys[:-1])
        conf_ = self[last_id]
        indexing = keys[-1] if isinstance(conf_, dict) else int(keys[-1])
        conf_[indexing] = config
        self.ref_resolver.reset()
        return

    def get(self, id: str = "", default: Optional[Any] = None):
        """
        Get the config by id.

        Args:
            id: id to specify the expected position. See also :py:meth:`__getitem__`.
            default: default value to return if the specified ``id`` is invalid.

        """
        try:
            return self[id]
        except (KeyError, IndexError):
            return default

    def set(self, config: Any, id: str = ""):
        """
        Set config by ``id``.

        Args:
            config: config to set at location ``id``.
            id: id to specify the expected position. See also :py:meth:`__setitem__`.

        """
        self[id] = config

    def update(self, pairs: Dict[str, Any]):
        """
        Set the ``id`` and the corresponding config content in pairs, see also :py:meth:`__setitem__`.
        For example, ``parser.update({"train#epoch": 100, "train#lr": 0.02})``

        Args:
            pairs: dictionary of `id` and config pairs.

        """
        for k, v in pairs.items():
            self[k] = v

    def __contains__(self, id: Union[str, int]) -> bool:
        """
        Returns True if `id` is stored in this configuration.

        Args:
            id: id to specify the expected position. See also :py:meth:`__getitem__`.
        """
        try:
            _ = self[id]
            return True
        except KeyError:
            return False

    def parse(self, reset: bool = True):
        """
        Recursively resolve `self.config` to replace the macro tokens with target content.
        Then recursively parse the config source, add every item as ``ConfigItem`` to the reference resolver.

        Args:
            reset: whether to reset the ``reference_resolver`` before parsing. Defaults to `True`.

        """
        if reset:
            self.ref_resolver.reset()
        self.resolve_macro_and_relative_ids()
        self._do_parse(config=self.get())

    def get_parsed_content(self, id: str = "", **kwargs):
        """
        Get the parsed result of ``ConfigItem`` with the specified ``id``.

            - If the item is ``ConfigComponent`` and ``instantiate=True``, the result is the instance.
            - If the item is ``ConfigExpression`` and ``eval_expr=True``, the result is the evaluated output.
            - Else, the result is the configuration content of `ConfigItem`.

        Args:
            id: id of the ``ConfigItem``, ``"#"`` in id are interpreted as special characters to
                go one level further into the nested structures.
                Use digits indexing from "0" for list or other strings for dict.
                For example: ``"xform#5"``, ``"net#channels"``. ``""`` indicates the entire ``self.config``.
            kwargs: additional keyword arguments to be passed to ``_resolve_one_item``.
                Currently support ``lazy`` (whether to retain the current config cache, default to `True`),
                ``instantiate`` (whether to instantiate the `ConfigComponent`, default to `True`) and
                ``eval_expr`` (whether to evaluate the `ConfigExpression`, default to `True`), ``default``
                (the default config item if the `id` is not in the config content).

        """
        if not self.ref_resolver.is_resolved():
            # not parsed the config source yet, parse it
            self.parse(reset=True)
        elif not kwargs.get("lazy", True):
            self.parse(reset=not kwargs.get("lazy", True))
        return self.ref_resolver.get_resolved_content(id=id, **kwargs)

    def read_meta(self, f: Union[PathLike, Sequence[PathLike], Dict], **kwargs):
        """
        Read the metadata from specified JSON or YAML file.
        The metadata as a dictionary will be stored at ``self.config["_meta_"]``.

        Args:
            f: filepath of the metadata file, the content must be a dictionary,
                if providing a list of files, will merge the content of them.
                if providing a dictionary directly, use it as metadata.
            kwargs: other arguments for ``json.load`` or ``yaml.safe_load``, depends on the file format.

        """
        self.set(self.load_config_files(f, **kwargs), self.meta_key)

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
        content = {self.meta_key: self.get(self.meta_key, {})}
        content.update(self.load_config_files(f, **kwargs))
        self.set(config=content)

    def _do_resolve(self, config: Any, id: str = ""):
        """
        Recursively resolve `self.config` to replace the relative ids with absolute ids, for example,
        `@##A` means `A` in the upper level. and replace the macro tokens with target content,
        The macro tokens start with "%", can be from another structured file, like:
        ``"%default_net"``, ``"%/data/config.json#net"``.
        Note that the macro replacement doesn't support recursive macro tokens.

        Args:
            config: input config file to resolve.
            id: id of the ``ConfigItem``, ``"#"`` in id are interpreted as special characters to
                go one level further into the nested structures.
                Use digits indexing from "0" for list or other strings for dict.
                For example: ``"xform#5"``, ``"net#channels"``. ``""`` indicates the entire ``self.config``.

        """
        if isinstance(config, (dict, list)):
            for k, v in enumerate(config) if isinstance(config, list) else config.items():
                sub_id = f"{id}{ID_SEP_KEY}{k}" if id != "" else k
                config[k] = self._do_resolve(v, sub_id)
        if isinstance(config, str):
            config = self.resolve_relative_ids(id, config)
            if config.startswith(MACRO_KEY):
                path, ids = ConfigParser.split_path_id(config[len(MACRO_KEY) :])
                parser = ConfigParser(config=self.get() if not path else ConfigParser.load_config_file(path))
                return parser[ids]
        return config

    def resolve_macro_and_relative_ids(self):
        """
        Recursively resolve `self.config` to replace the relative ids with absolute ids, for example,
        `@##A` means `A` in the upper level. and replace the macro tokens with target content,
        The macro tokens are marked as starting with "%", can be from another structured file, like:
        ``"%default_net"``, ``"%/data/config.json#net"``.

        """
        self.set(self._do_resolve(config=deepcopy(self.get())))

    def _do_parse(self, config, id: str = ""):
        """
        Recursively parse the nested data in config source, add every item as `ConfigItem` to the resolver.

        Args:
            config: config source to parse.
            id: id of the ``ConfigItem``, ``"#"`` in id are interpreted as special characters to
                go one level further into the nested structures.
                Use digits indexing from "0" for list or other strings for dict.
                For example: ``"xform#5"``, ``"net#channels"``. ``""`` indicates the entire ``self.config``.

        """
        if isinstance(config, (dict, list)):
            for k, v in enumerate(config) if isinstance(config, list) else config.items():
                sub_id = f"{id}{ID_SEP_KEY}{k}" if id != "" else k
                self._do_parse(config=v, id=sub_id)

        # copy every config item to make them independent and add them to the resolver
        item_conf = deepcopy(config)
        if ConfigComponent.is_instantiable(item_conf):
            self.ref_resolver.add_item(ConfigComponent(config=item_conf, id=id, locator=self.locator))
        elif ConfigExpression.is_expression(item_conf):
            self.ref_resolver.add_item(ConfigExpression(config=item_conf, id=id, globals=self.globals))
        else:
            self.ref_resolver.add_item(ConfigItem(config=item_conf, id=id))

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
    def load_config_files(cls, files: Union[PathLike, Sequence[PathLike], dict], **kwargs) -> Dict:
        """
        Load config files into a single config dict.
        The latter config file in the list will override or add the former config file.
        ``"#"`` in the config keys are interpreted as special characters to go one level
        further into the nested structures.

        Args:
            files: path of target files to load, supported postfixes: `.json`, `.yml`, `.yaml`.
            kwargs: other arguments for ``json.load`` or ```yaml.safe_load``, depends on the file format.
        """
        if isinstance(files, dict):  # already a config dict
            return files
        parser = ConfigParser(config={})
        for i in ensure_tuple(files):
            for k, v in (cls.load_config_file(i, **kwargs)).items():
                parser[k] = v
        return parser.get()  # type: ignore

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
        result = re.compile(rf"({cls.suffix_match}(?=(?:{ID_SEP_KEY}.*)|$))", re.IGNORECASE).findall(src)
        if not result:
            return "", src  # the src is a pure id
        path_name = result[0][0]  # at most one path_name
        _, ids = src.rsplit(path_name, 1)
        return path_name, ids[len(ID_SEP_KEY) :] if ids.startswith(ID_SEP_KEY) else ""

    @classmethod
    def resolve_relative_ids(cls, id: str, value: str) -> str:
        """
        To simplify the reference or macro tokens ID in the nested config content, it's available to use
        relative ID name which starts with the `ID_SEP_KEY`, for example, "@#A" means `A` in the same level,
        `@##A` means `A` in the upper level.
        It resolves the relative ids to absolute ids. For example, if the input data is:

        .. code-block:: python

            {
                "A": 1,
                "B": {"key": "@##A", "value1": 2, "value2": "%#value1", "value3": [3, 4, "@#1"]},
            }

        It will resolve `B` to `{"key": "@A", "value1": 2, "value2": "%B#value1", "value3": [3, 4, "@B#value3#1"]}`.

        Args:
            id: id name for current config item to compute relative id.
            value: input value to resolve relative ids.

        """
        # get the prefixes like: "@####", "%###", "@#"
        prefixes = sorted(set().union(cls.relative_id_prefix.findall(value)), reverse=True)
        current_id = id.split(ID_SEP_KEY)

        for p in prefixes:
            sym = ID_REF_KEY if ID_REF_KEY in p else MACRO_KEY
            length = p[len(sym) :].count(ID_SEP_KEY)
            if length > len(current_id):
                raise ValueError(f"the relative id in `{value}` is out of the range of config content.")
            if length == len(current_id):
                new = ""  # root id is `""`
            else:
                new = ID_SEP_KEY.join(current_id[:-length]) + ID_SEP_KEY
            value = value.replace(p, sym + new)
        return value

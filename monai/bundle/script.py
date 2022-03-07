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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Union

from monai.bundle.config_parser import ConfigParser
from monai.bundle.utils import id_value_str_to_dict, load_config_file, load_config_file_content
from monai.utils import ensure_tuple, optional_import

fire, _ = optional_import("fire")


class Script(ABC):
    """
    Base class for typical config based scripts in the bundle.
    Typical usage examples:

    1. Execute the `run` API with other CLI tools, take `fire` for example:
    `python -m fire monai.bundle run --meta_file=<meta path> --config_file=<config path> --target=trainer`

    2. Execute this module as CLI entry based on `fire`:
    `python -m monai.bundle.scripts run --meta_file=<meta path> --config_file=<config path> --target=trainer`

    3. Override some config values at runtime, set `override` as a dict:
    `python -m monai.bundle.scripts run --override={"'net#<args>#ndims'": 2} ...`

    4. Override some config values at runtime, set `override` as a string:
    `python -m monai.bundle.scripts run --override="{net#<args>#ndims: 2}" ...`

    5. Override some config values with another config file:
    `python -m monai.bundle.scripts run --override={"'net#<args>'": "'<file>/data/other.json'"} ...`

    6. Override some config values with part content of another config file:
    `python -m monai.bundle.scripts run --override={"'net#<args>'": "'<file>/data/other.json#net_arg'"} ...`

    7. Set default args of `run` in a JSON / YAML file, help to record and simplify the command line.
    Other args still can override the default args at runtime:
    `python -m monai.bundle.scripts run --args_file="'/data/args.json'" --config_file=<config path>`

    Args:
        meta_file: filepath of the metadata file, if None, must provide it in `arg_file`.
            if providing a list of files, wil merge the content of them.
        config_file: filepath of the config file, if None, must provide it in `arg_file`.
            if providing a list of files, wil merge the content of them.
        override: override above config content with specified `id` and `value` pairs.
            it can also be used to provide default value for placeholders. for example:
            put a placeholder `"data": "@runtime_value"` in the config, then define
            `runtime_value` in `override`.
            it also supports a string representing a dict, like: "{'AA#BB': 123}", usually from command line.
        args_file: to avoid providing same args every time running the program, it supports
            to put the args as a dictionary in a JSON or YAML file.
        kwargs: other args that can set in `args_file` and override at runtime.

    """
    def __init__(
        self,
        meta_file: Optional[Union[str, Sequence[str]]] = None,
        config_file: Optional[Union[str, Sequence[str]]] = None,
        override: Optional[Union[Dict, str]] = None,
        args_file: Optional[str] = None,
        **kwargs,
    ):
        if isinstance(override, str):
            # if override is a string representing a dict (usually from command line), convert it to dict
            override = id_value_str_to_dict(override)
        self.args = self._update_default_args(
            args=args_file, meta_file=meta_file, config_file=config_file, override=override, **kwargs,
        )
        self.parser = self.parse_config_file(
            config_file=self.args["config_file"], meta_file=self.args["meta_file"], override=self.args["override"]
        )

    def _update_default_args(self, args: Optional[Union[str, Dict]] = None, **kwargs) -> Dict:
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
            if isinstance(v, dict) and isinstance(args_.get(k), dict):
                args_[k] = self._update_default_args(args_[k], **v)
            else:
                args_[k] = v
        return args_

    def parse_config_file(
        self,
        config_file: Union[str, Sequence[str]],
        meta_file: Union[str, Sequence[str]],
        override: Optional[Dict] = None,
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

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any):
        """
        Execute task specific script.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class Run(Script):
    """
    Specify metadata file and config file to run a regular training or evaluation program.
    It's used to execute most of the supervised training, evaluation or inference cases.

    Args:
        meta_file: filepath of the metadata file, if None, must provide it in `arg_file`.
            if providing a list of files, wil merge the content of them.
        config_file: filepath of the config file, if None, must provide it in `arg_file`.
            if providing a list of files, wil merge the content of them.
        override: override above config content with specified `id` and `value` pairs.
            it can also be used to provide default value for placeholders. for example:
            put a placeholder `"data": "@runtime_value"` in the config, then define
            `runtime_value` in `override`.
            it also supports a string representing a dict, like: "{'AA#BB': 123}", usually from command line.
        target: ID name of the target workflow, it must have the `run` method, follow MONAI `BaseWorkflow`.
            if None, must provide it in `arg_file`.
        args_file: to avoid providing same args every time running the program, it supports
            to put the args as a dictionary in a JSON or YAML file.
        kwargs: other args that can set in `args_file` and override at runtime.

    """
    def __init__(
        self,
        meta_file: Optional[Union[str, Sequence[str]]] = None,
        config_file: Optional[Union[str, Sequence[str]]] = None,
        override: Optional[Union[Dict, str]] = None,
        target: Optional[str] = None,
        args_file: Optional[str] = None,
    ):
        super().__init__(
            meta_file=meta_file, config_file=config_file, override=override, args_file=args_file, target=target,
        )

    def __call__(self):
        # get expected workflow to run
        workflow = self.parser.get_parsed_content(id=self.args["target"])
        workflow.run()


if __name__ == "__main__":
    fire.Fire()

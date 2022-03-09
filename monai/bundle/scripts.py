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

from typing import Dict, Optional, Sequence, Union

from monai.bundle.config_parser import ConfigParser
from monai.bundle.config_reader import ConfigReader
from monai.bundle.utils import update_default_args


def run(
    meta_file: Optional[Union[str, Sequence[str]]] = None,
    config_file: Optional[Union[str, Sequence[str]]] = None,
    override: Optional[Union[Dict, str]] = None,
    target: Optional[str] = None,
    args_file: Optional[str] = None,
    **kwargs,
):
    """
    Specify metadata file and config file to run a regular training or evaluation program.
    It's used to execute most of the supervised training, evaluation or inference cases.

    Typical usage examples:

    .. code-block:: bash

        # Execute this module as CLI entry:
        python -m monai.bundle run --meta_file=<meta path> --config_file=<config path> --target=trainer

        # Override some config values at runtime, set `override` as a dict:
        python -m monai.bundle run --override='{"net#<args>#ndims": 2}' ...

        # Override some config values at runtime:
        python -m monai.bundle run --"net#<args>#input_chns" 1 ...

        # Override some config values with another config file:
        python -m monai.bundle run --override='{"net#<args>": "%/data/other.json"}' ...

        # Override some config values with part content of another config file:
        python -m monai.bundle run --override='{"net#<args>": "%/data/other.json#net_arg"}' ...

        # Set default args of `run` in a JSON / YAML file, help to record and simplify the command line.
        # Other args still can override the default args at runtime:
        python -m monai.bundle run --args_file="'/workspace/data/args.json'" --config_file=<config path>

    Args:
        meta_file: filepath of the metadata file, if `None`, must provide it in `args_file`.
            if providing a list of files, wil merge the content of them.
        config_file: filepath of the config file, if `None`, must provide it in `args_file`.
            if providing a list of files, wil merge the content of them.
        override: override config content with specified `id` and `value` pairs.
            it can also be used to provide default value for placeholders. for example:
            put a placeholder `"data": "@runtime_value"` in the config, then define `runtime_value` in `override`.
            it also supports a string representing a dict, like: "{'AA#BB': 123}", usually from command line.
        target: ID name of the target workflow, it must have the `run` method, follow MONAI `BaseWorkflow`.
            if None, must provide it in `arg_file`.
        args_file: to avoid providing same args every time running the program, it supports
            to put the args as a dictionary in a JSON or YAML file.
        kwargs: additional id-value pairs to override the config content.

    """
    k_v = zip(
        ["meta_file", "config_file", "override", "target", "args_file"],
        [meta_file, config_file, override, target, args_file],
    )
    input_args = {k: v for k, v in k_v if v is not None}
    _args = update_default_args(args=args_file, **input_args)
    for k in ("meta_file", "config_file", "target"):
        if k not in _args:
            raise ValueError(f"{k} is required.")

    reader = ConfigReader()
    reader.read_config(f=_args["config_file"])
    reader.read_meta(f=_args["meta_file"])

    parser = ConfigParser(reader.get())

    override = _args.get("override", {})
    if isinstance(override, dict):
        override.update(kwargs)
    if override and isinstance(override, dict):
        for k, v in override.items():
            parser[k] = v

    # get expected workflow to run
    workflow = parser.get_parsed_content(id=_args["target"])
    workflow.run()

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

from monai.bundle.utils import parse_config_file, update_default_args
from monai.utils import optional_import

fire, _ = optional_import("fire")


def run(
    meta_file: Optional[Union[str, Sequence[str]]] = None,
    config_file: Optional[Union[str, Sequence[str]]] = None,
    override: Optional[Union[Dict, str]] = None,
    target: Optional[str] = None,
    args_file: Optional[str] = None,
):
    """
    Specify metadata file and config file to run a regular training or evaluation program.
    It's used to execute most of the supervised training, evaluation or inference cases.

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
        target: ID name of the target workflow, it must have the `run` method, follow MONAI `BaseWorkflow`.
            if None, must provide it in `arg_file`.
        args_file: to avoid providing same args every time running the program, it supports
            to put the args as a dictionary in a JSON or YAML file.

    """

    kwargs = {}
    for k, v in {"meta_file": meta_file, "config_file": config_file, "override": override, "target": target}.items():
        if v is not None:
            # skip None args
            kwargs[k] = v
    args = update_default_args(args=args_file, **kwargs)

    config_parser = parse_config_file(
        config_file=args["config_file"], meta_file=args["meta_file"], override=args.get("override")
    )
    # get expected workflow to run
    workflow = config_parser.get_parsed_content(id=args["target"])
    workflow.run()


if __name__ == "__main__":
    fire.Fire()

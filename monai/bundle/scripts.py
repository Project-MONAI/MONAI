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

import pprint
from typing import Dict, Optional, Sequence, Union

from monai.bundle.config_parser import ConfigParser


def _update_default_args(args: Optional[Union[str, Dict]] = None, **kwargs) -> Dict:
    """
    Update the `args` with the input `kwargs`.
    For dict data, recursively update the content based on the keys.

    Args:
        args: source args to update.
        kwargs: destination args to update.

    """
    args_: Dict = args if isinstance(args, dict) else {}  # type: ignore
    if isinstance(args, str):
        # args are defined in a structured file
        args_ = ConfigParser.load_config_file(args)

    # recursively update the default args with new args
    for k, v in kwargs.items():
        args_[k] = _update_default_args(args_[k], **v) if isinstance(v, dict) and isinstance(args_.get(k), dict) else v
    return args_


def run(
    meta_file: Optional[Union[str, Sequence[str]]] = None,
    config_file: Optional[Union[str, Sequence[str]]] = None,
    target_id: Optional[str] = None,
    args_file: Optional[str] = None,
    **override,
):
    """
    Specify `meta_file` and `config_file` to run monai bundle components and workflows.

    Typical usage examples:

    .. code-block:: bash

        # Execute this module as a CLI entry:
        python -m monai.bundle run --meta_file <meta path> --config_file <config path> --target_id trainer

        # Override config values at runtime by specifying the component id and its new value:
        python -m monai.bundle run --net#input_chns 1 ...

        # Override config values with another config file `/path/to/another.json`:
        python -m monai.bundle run --net %/path/to/another.json ...

        # Override config values with part content of another config file:
        python -m monai.bundle run --net %/data/other.json#net_arg ...

        # Set default args of `run` in a JSON / YAML file, help to record and simplify the command line.
        # Other args still can override the default args at runtime:
        python -m monai.bundle run --args_file "/workspace/data/args.json" --config_file <config path>

    Args:
        meta_file: filepath of the metadata file, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        config_file: filepath of the config file, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        target_id: ID name of the target component or workflow, it must have a `run` method.
        args_file: a JSON or YAML file to provide default values for `meta_file`, `config_file`,
            `target_id` and override pairs. so that the command line inputs can be simplified.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--net#input_chns 42``.

    """
    k_v = zip(["meta_file", "config_file", "target_id"], [meta_file, config_file, target_id])
    for k, v in k_v:
        if v is not None:
            override[k] = v

    full_kv = zip(
        ("meta_file", "config_file", "target_id", "args_file", "override"),
        (meta_file, config_file, target_id, args_file, override),
    )
    print("\n--- input summary of monai.bundle.scripts.run ---")
    for name, val in full_kv:
        print(f"> {name}: {pprint.pformat(val)}")
    print("---\n\n")

    _args = _update_default_args(args=args_file, **override)
    for k in ("meta_file", "config_file"):
        if k not in _args:
            raise ValueError(f"{k} is required for 'monai.bundle run'.\n{run.__doc__}")

    parser = ConfigParser()
    parser.read_config(f=_args.pop("config_file"))
    parser.read_meta(f=_args.pop("meta_file"))
    id = _args.pop("target_id", "")

    # the rest key-values in the args are to override config content
    for k, v in _args.items():
        parser[k] = v

    workflow = parser.get_parsed_content(id=id)
    if not hasattr(workflow, "run"):
        raise ValueError(f"The parsed workflow {type(workflow)} does not have a `run` method.\n{run.__doc__}")
    workflow.run()

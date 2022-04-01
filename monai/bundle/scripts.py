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

import ast
import json
import pprint
import re
from logging.config import fileConfig
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch.cuda import is_available

from monai.apps.utils import download_url, get_logger
from monai.bundle.config_parser import ConfigParser
from monai.config import IgniteInfo, PathLike
from monai.data import save_net_with_metadata
from monai.networks import convert_to_torchscript, copy_model_state
from monai.utils import check_parent_dir, get_equivalent_dtype, min_version, optional_import

validate, _ = optional_import("jsonschema", name="validate")
ValidationError, _ = optional_import("jsonschema.exceptions", name="ValidationError")
Checkpoint, has_ignite = optional_import("ignite.handlers", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Checkpoint")

logger = get_logger(module_name=__name__)


def _update_args(args: Optional[Union[str, Dict]] = None, ignore_none: bool = True, **kwargs) -> Dict:
    """
    Update the `args` with the input `kwargs`.
    For dict data, recursively update the content based on the keys.

    Args:
        args: source args to update.
        ignore_none: whether to ignore input args with None value, default to `True`.
        kwargs: destination args to update.

    """
    args_: Dict = args if isinstance(args, dict) else {}  # type: ignore
    if isinstance(args, str):
        # args are defined in a structured file
        args_ = ConfigParser.load_config_file(args)

    # recursively update the default args with new args
    for k, v in kwargs.items():
        if ignore_none and v is None:
            continue
        if isinstance(v, dict) and isinstance(args_.get(k), dict):
            args_[k] = _update_args(args_[k], ignore_none, **v)
        else:
            args_[k] = v
    return args_


def _pop_args(src: Dict, *args, **kwargs):
    """
    Pop args from the `src` dictionary based on specified keys in `args` and (key, default value) pairs in `kwargs`.

    """
    return tuple([src.pop(i) for i in args] + [src.pop(k, v) for k, v in kwargs.items()])


def _log_input_summary(tag, args: Dict):
    logger.info(f"--- input summary of monai.bundle.scripts.{tag} ---")
    for name, val in args.items():
        logger.info(f"> {name}: {pprint.pformat(val)}")
    logger.info("---\n\n")


def _get_var_names(expr: str):
    """
    Parse the expression and discover what variables are present in it based on ast module.

    Args:
        expr: source expression to parse.

    """
    tree = ast.parse(expr)
    return [m.id for m in ast.walk(tree) if isinstance(m, ast.Name)]


def _get_fake_spatial_shape(shape: Sequence[Union[str, int]], p: int = 1, n: int = 1, any: int = 1) -> Tuple:
    """
    Get spatial shape for fake data according to the specified shape pattern.
    It supports `int` number and `string` with formats like: "32", "32 * n", "32 ** p", "32 ** p *n".

    Args:
        shape: specified pattern for the spatial shape.
        p: power factor to generate fake data shape if dim of expected shape is "x**p", default to 1.
        p: multiply factor to generate fake data shape if dim of expected shape is "x*n", default to 1.
        any: specified size to generate fake data shape if dim of expected shape is "*", default to 1.

    """
    ret = []
    for i in shape:
        if isinstance(i, int):
            ret.append(i)
        elif isinstance(i, str):
            if i == "*":
                ret.append(any)
            else:
                for c in _get_var_names(i):
                    if c not in ["p", "n"]:
                        raise ValueError(f"only support variables 'p' and 'n' so far, but got: {c}.")
                ret.append(eval(i, {"p": p, "n": n}))
        else:
            raise ValueError(f"spatial shape items must be int or string, but got: {type(i)} {i}.")
    return tuple(ret)


def run(
    runner_id: Optional[str] = None,
    meta_file: Optional[Union[str, Sequence[str]]] = None,
    config_file: Optional[Union[str, Sequence[str]]] = None,
    logging_file: Optional[str] = None,
    args_file: Optional[str] = None,
    **override,
):
    """
    Specify `meta_file` and `config_file` to run monai bundle components and workflows.

    Typical usage examples:

    .. code-block:: bash

        # Execute this module as a CLI entry:
        python -m monai.bundle run trainer --meta_file <meta path> --config_file <config path>

        # Override config values at runtime by specifying the component id and its new value:
        python -m monai.bundle run trainer --net#input_chns 1 ...

        # Override config values with another config file `/path/to/another.json`:
        python -m monai.bundle run evaluator --net %/path/to/another.json ...

        # Override config values with part content of another config file:
        python -m monai.bundle run trainer --net %/data/other.json#net_arg ...

        # Set default args of `run` in a JSON / YAML file, help to record and simplify the command line.
        # Other args still can override the default args at runtime:
        python -m monai.bundle run --args_file "/workspace/data/args.json" --config_file <config path>

    Args:
        runner_id: ID name of the runner component or workflow, it must have a `run` method. Defaults to ``""``.
        meta_file: filepath of the metadata file, if it is a list of file paths, the content of them will be merged.
        config_file: filepath of the config file, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        logging_file: config file for `logging` module in the program, default to `None`. for more details:
            https://docs.python.org/3/library/logging.config.html#logging.config.fileConfig.
        args_file: a JSON or YAML file to provide default values for `runner_id`, `meta_file`,
            `config_file`, `logging`, and override pairs. so that the command line inputs can be simplified.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--net#input_chns 42``.

    """

    _args = _update_args(
        args=args_file,
        runner_id=runner_id,
        meta_file=meta_file,
        config_file=config_file,
        logging_file=logging_file,
        **override,
    )
    if "config_file" not in _args:
        raise ValueError(f"`config_file` is required for 'monai.bundle run'.\n{run.__doc__}")
    _log_input_summary(tag="run", args=_args)
    config_file_, meta_file_, runner_id_, logging_file_ = _pop_args(
        _args, "config_file", meta_file=None, runner_id="", logging_file=None
    )
    if logging_file_ is not None:
        logger.info(f"set logging properties based on config: {logging_file_}.")
        fileConfig(logging_file_, disable_existing_loggers=False)

    parser = ConfigParser()
    parser.read_config(f=config_file_)
    if meta_file_ is not None:
        parser.read_meta(f=meta_file_)

    # the rest key-values in the _args are to override config content
    for k, v in _args.items():
        parser[k] = v

    workflow = parser.get_parsed_content(id=runner_id_)
    if not hasattr(workflow, "run"):
        raise ValueError(
            f"The parsed workflow {type(workflow)} (id={runner_id_}) does not have a `run` method.\n{run.__doc__}"
        )
    return workflow.run()


def verify_metadata(
    meta_file: Optional[Union[str, Sequence[str]]] = None,
    filepath: Optional[PathLike] = None,
    create_dir: Optional[bool] = None,
    hash_val: Optional[str] = None,
    hash_type: Optional[str] = None,
    args_file: Optional[str] = None,
    **kwargs,
):
    """
    Verify the provided `metadata` file based on the predefined `schema`.
    `metadata` content must contain the `schema` field for the URL of schema file to download.
    The schema standard follows: http://json-schema.org/.

    Args:
        meta_file: filepath of the metadata file to verify, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        filepath: file path to store the downloaded schema.
        create_dir: whether to create directories if not existing, default to `True`.
        hash_val: if not None, define the hash value to verify the downloaded schema file.
        hash_type: if not None, define the hash type to verify the downloaded schema file. Defaults to "md5".
        args_file: a JSON or YAML file to provide default values for all the args in this function.
            so that the command line inputs can be simplified.
        kwargs: other arguments for `jsonschema.validate()`. for more details:
            https://python-jsonschema.readthedocs.io/en/stable/validate/#jsonschema.validate.

    """

    _args = _update_args(
        args=args_file,
        meta_file=meta_file,
        filepath=filepath,
        create_dir=create_dir,
        hash_val=hash_val,
        hash_type=hash_type,
        **kwargs,
    )
    _log_input_summary(tag="verify_metadata", args=_args)
    filepath_, meta_file_, create_dir_, hash_val_, hash_type_ = _pop_args(
        _args, "filepath", "meta_file", create_dir=True, hash_val=None, hash_type="md5"
    )

    check_parent_dir(path=filepath_, create_dir=create_dir_)
    metadata = ConfigParser.load_config_files(files=meta_file_)
    url = metadata.get("schema")
    if url is None:
        raise ValueError("must provide the `schema` field in the metadata for the URL of schema file.")
    download_url(url=url, filepath=filepath_, hash_val=hash_val_, hash_type=hash_type_, progress=True)
    schema = ConfigParser.load_config_file(filepath=filepath_)

    try:
        # the rest key-values in the _args are for `validate` API
        validate(instance=metadata, schema=schema, **_args)
    except ValidationError as e:
        # as the error message is very long, only extract the key information
        logger.info(re.compile(r".*Failed validating", re.S).findall(str(e))[0] + f" against schema `{url}`.")
        return
    logger.info("metadata is verified with no error.")


def verify_net_in_out(
    net_id: Optional[str] = None,
    meta_file: Optional[Union[str, Sequence[str]]] = None,
    config_file: Optional[Union[str, Sequence[str]]] = None,
    device: Optional[str] = None,
    p: Optional[int] = None,
    n: Optional[int] = None,
    any: Optional[int] = None,
    args_file: Optional[str] = None,
    **override,
):
    """
    Verify the input and output data shape and data type of network defined in the metadata.
    Will test with fake Tensor data according to the required data shape in `metadata`.

    Typical usage examples:

    .. code-block:: bash

        python -m monai.bundle verify_net_in_out network --meta_file <meta path> --config_file <config path>

    Args:
        net_id: ID name of the network component to verify, it must be `torch.nn.Module`.
        meta_file: filepath of the metadata file to get network args, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        config_file: filepath of the config file to get network definition, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        device: target device to run the network forward computation, if None, prefer to "cuda" if existing.
        p: power factor to generate fake data shape if dim of expected shape is "x**p", default to 1.
        n: multiply factor to generate fake data shape if dim of expected shape is "x*n", default to 1.
        any: specified size to generate fake data shape if dim of expected shape is "*", default to 1.
        args_file: a JSON or YAML file to provide default values for `net_id`, `meta_file`, `config_file`,
            `device`, `p`, `n`, `any`, and override pairs. so that the command line inputs can be simplified.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--_meta#network_data_format#inputs#image#num_channels 3``.

    """

    _args = _update_args(
        args=args_file,
        net_id=net_id,
        meta_file=meta_file,
        config_file=config_file,
        device=device,
        p=p,
        n=n,
        any=any,
        **override,
    )
    _log_input_summary(tag="verify_net_in_out", args=_args)
    config_file_, meta_file_, net_id_, device_, p_, n_, any_ = _pop_args(
        _args, "config_file", "meta_file", net_id="", device="cuda:0" if is_available() else "cpu", p=1, n=1, any=1
    )

    parser = ConfigParser()
    parser.read_config(f=config_file_)
    parser.read_meta(f=meta_file_)

    # the rest key-values in the _args are to override config content
    for k, v in _args.items():
        parser[k] = v

    try:
        key: str = net_id_  # mark the full id when KeyError
        net = parser.get_parsed_content(key).to(device_)
        key = "_meta_#network_data_format#inputs#image#num_channels"
        input_channels = parser[key]
        key = "_meta_#network_data_format#inputs#image#spatial_shape"
        input_spatial_shape = tuple(parser[key])
        key = "_meta_#network_data_format#inputs#image#dtype"
        input_dtype = get_equivalent_dtype(parser[key], torch.Tensor)
        key = "_meta_#network_data_format#outputs#pred#num_channels"
        output_channels = parser[key]
        key = "_meta_#network_data_format#outputs#pred#dtype"
        output_dtype = get_equivalent_dtype(parser[key], torch.Tensor)
    except KeyError as e:
        raise KeyError(f"Failed to verify due to missing expected key in the config: {key}.") from e

    net.eval()
    with torch.no_grad():
        spatial_shape = _get_fake_spatial_shape(input_spatial_shape, p=p_, n=n_, any=any_)  # type: ignore
        test_data = torch.rand(*(1, input_channels, *spatial_shape), dtype=input_dtype, device=device_)
        output = net(test_data)
        if output.shape[1] != output_channels:
            raise ValueError(f"output channel number `{output.shape[1]}` doesn't match: `{output_channels}`.")
        if output.dtype != output_dtype:
            raise ValueError(f"dtype of output data `{output.dtype}` doesn't match: {output_dtype}.")
    logger.info("data shape of network is verified with no error.")


def ckpt_export(
    net_id: Optional[str] = None,
    filepath: Optional[PathLike] = None,
    ckpt_file: Optional[str] = None,
    meta_file: Optional[Union[str, Sequence[str]]] = None,
    config_file: Optional[Union[str, Sequence[str]]] = None,
    key_in_ckpt: Optional[str] = None,
    args_file: Optional[str] = None,
    **override,
):
    """
    Export the model checkpoint to the given filepath with metadata and config included as JSON files.

    Typical usage examples:

    .. code-block:: bash

        python -m monai.bundle export network --filepath <export path> --ckpt_file <checkpoint path> ...

    Args:
        net_id: ID name of the network component in the config, it must be `torch.nn.Module`.
        filepath: filepath to export, if filename has no extension it becomes `.ts`.
        ckpt_file: filepath of the model checkpoint to load.
        meta_file: filepath of the metadata file, if it is a list of file paths, the content of them will be merged.
        config_file: filepath of the config file, if `None`, must be provided in `args_file`.
            if it is a list of file paths, the content of them will be merged.
        key_in_ckpt: for nested checkpoint like `{"model": XXX, "optimizer": XXX, ...}`, specify the key of model
            weights. if not nested checkpoint, no need to set.
        args_file: a JSON or YAML file to provide default values for `meta_file`, `config_file`,
            `net_id` and override pairs. so that the command line inputs can be simplified.
        override: id-value pairs to override or add the corresponding config content.
            e.g. ``--_meta#network_data_format#inputs#image#num_channels 3``.

    """
    _args = _update_args(
        args=args_file,
        net_id=net_id,
        filepath=filepath,
        meta_file=meta_file,
        config_file=config_file,
        ckpt_file=ckpt_file,
        key_in_ckpt=key_in_ckpt,
        **override,
    )
    _log_input_summary(tag="export", args=_args)
    filepath_, ckpt_file_, config_file_, net_id_, meta_file_, key_in_ckpt_ = _pop_args(
        _args, "filepath", "ckpt_file", "config_file", net_id="", meta_file=None, key_in_ckpt=""
    )

    parser = ConfigParser()

    parser.read_config(f=config_file_)
    if meta_file_ is not None:
        parser.read_meta(f=meta_file_)

    # the rest key-values in the _args are to override config content
    for k, v in _args.items():
        parser[k] = v

    net = parser.get_parsed_content(net_id_)
    if has_ignite:
        # here we use ignite Checkpoint to support nested weights and be compatible with MONAI CheckpointSaver
        Checkpoint.load_objects(to_load={key_in_ckpt_: net}, checkpoint=ckpt_file_)
    else:
        copy_model_state(dst=net, src=ckpt_file_ if key_in_ckpt_ == "" else ckpt_file_[key_in_ckpt_])

    # convert to TorchScript model and save with meta data, config content
    net = convert_to_torchscript(model=net)

    save_net_with_metadata(
        jit_obj=net,
        filename_prefix_or_stream=filepath_,
        include_config_vals=False,
        append_timestamp=False,
        meta_values=parser.get().pop("_meta_", None),
        more_extra_files={"config": json.dumps(parser.get()).encode()},
    )
    logger.info(f"exported to TorchScript file: {filepath_}.")

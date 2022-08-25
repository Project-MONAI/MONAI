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
import os
import pprint
import re
import warnings
from logging.config import fileConfig
from pathlib import Path
from shutil import copyfile
from textwrap import dedent
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.cuda import is_available

from monai.apps.utils import _basename, download_url, extractall, get_logger
from monai.bundle.config_item import ConfigComponent
from monai.bundle.config_parser import ConfigParser
from monai.bundle.utils import DEFAULT_INFERENCE, DEFAULT_METADATA
from monai.config import IgniteInfo, PathLike
from monai.data import load_net_with_metadata, save_net_with_metadata
from monai.networks import convert_to_torchscript, copy_model_state, get_state_dict, save_state
from monai.utils import check_parent_dir, get_equivalent_dtype, min_version, optional_import
from monai.utils.misc import ensure_tuple

validate, _ = optional_import("jsonschema", name="validate")
ValidationError, _ = optional_import("jsonschema.exceptions", name="ValidationError")
Checkpoint, has_ignite = optional_import("ignite.handlers", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Checkpoint")
requests_get, has_requests = optional_import("requests", name="get")

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
    args_: Dict = args if isinstance(args, dict) else {}
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


def _get_git_release_url(repo_owner: str, repo_name: str, tag_name: str, filename: str):
    return f"https://github.com/{repo_owner}/{repo_name}/releases/download/{tag_name}/{filename}"


def _download_from_github(repo: str, download_path: Path, filename: str, progress: bool = True):
    if len(repo.split("/")) != 3:
        raise ValueError("if source is `github`, repo should be in the form of `repo_owner/repo_name/release_tag`.")
    repo_owner, repo_name, tag_name = repo.split("/")
    if ".zip" not in filename:
        filename += ".zip"
    url = _get_git_release_url(repo_owner, repo_name, tag_name=tag_name, filename=filename)
    filepath = download_path / f"{filename}"
    download_url(url=url, filepath=filepath, hash_val=None, progress=progress)
    extractall(filepath=filepath, output_dir=download_path, has_base=True)


def _process_bundle_dir(bundle_dir: Optional[PathLike] = None):
    if bundle_dir is None:
        get_dir, has_home = optional_import("torch.hub", name="get_dir")
        if has_home:
            bundle_dir = Path(get_dir()) / "bundle"
        else:
            raise ValueError("bundle_dir=None, but no suitable default directory computed. Upgrade Pytorch to 1.6+ ?")
    return Path(bundle_dir)


def download(
    name: Optional[str] = None,
    bundle_dir: Optional[PathLike] = None,
    source: str = "github",
    repo: str = "Project-MONAI/model-zoo/hosting_storage_v1",
    url: Optional[str] = None,
    progress: bool = True,
    args_file: Optional[str] = None,
):
    """
    download bundle from the specified source or url. The bundle should be a zip file and it
    will be extracted after downloading.
    This function refers to:
    https://pytorch.org/docs/stable/_modules/torch/hub.html

    Typical usage examples:

    .. code-block:: bash

        # Execute this module as a CLI entry, and download bundle:
        python -m monai.bundle download --name "bundle_name" --source "github" --repo "repo_owner/repo_name/release_tag"

        # Execute this module as a CLI entry, and download bundle via URL:
        python -m monai.bundle download --name "bundle_name" --url <url>

        # Set default args of `run` in a JSON / YAML file, help to record and simplify the command line.
        # Other args still can override the default args at runtime.
        # The content of the JSON / YAML file is a dictionary. For example:
        # {"name": "spleen", "bundle_dir": "download", "source": ""}
        # then do the following command for downloading:
        python -m monai.bundle download --args_file "args.json" --source "github"

    Args:
        name: bundle name. If `None` and `url` is `None`, it must be provided in `args_file`.
        bundle_dir: target directory to store the downloaded data.
            Default is `bundle` subfolder under `torch.hub.get_dir()`.
        source: storage location name. This argument is used when `url` is `None`.
            "github" is currently the only supported value.
        repo: repo name. This argument is used when `url` is `None`.
            If `source` is "github", it should be in the form of "repo_owner/repo_name/release_tag".
        url: url to download the data. If not `None`, data will be downloaded directly
            and `source` will not be checked.
            If `name` is `None`, filename is determined by `monai.apps.utils._basename(url)`.
        progress: whether to display a progress bar.
        args_file: a JSON or YAML file to provide default values for all the args in this function.
            so that the command line inputs can be simplified.

    """
    _args = _update_args(
        args=args_file, name=name, bundle_dir=bundle_dir, source=source, repo=repo, url=url, progress=progress
    )

    _log_input_summary(tag="download", args=_args)
    source_, repo_, progress_, name_, bundle_dir_, url_ = _pop_args(
        _args, "source", "repo", "progress", name=None, bundle_dir=None, url=None
    )

    bundle_dir_ = _process_bundle_dir(bundle_dir_)

    if url_ is not None:
        if name is not None:
            filepath = bundle_dir_ / f"{name}.zip"
        else:
            filepath = bundle_dir_ / f"{_basename(url_)}"
        download_url(url=url_, filepath=filepath, hash_val=None, progress=progress_)
        extractall(filepath=filepath, output_dir=bundle_dir_, has_base=True)
    elif source_ == "github":
        if name_ is None:
            raise ValueError(f"To download from source: Github, `name` must be provided, got {name_}.")
        _download_from_github(repo=repo_, download_path=bundle_dir_, filename=name_, progress=progress_)
    else:
        raise NotImplementedError(
            f"Currently only download from provided URL in `url` or Github is implemented, got source: {source_}."
        )


def _get_all_bundles_info(
    repo: str = "Project-MONAI/model-zoo",
    tag: str = "hosting_storage_v1",
):
    if has_requests:
        request_url = f"https://api.github.com/repos/{repo}/releases"
        resp = requests_get(request_url)
        resp.raise_for_status()
    else:
        raise ValueError("requests package is required, please install it.")
    releases_list = json.loads(resp.text)
    bundle_name_pattern = re.compile(r"_v\d*.")
    bundles_info: Dict = {}

    for release in releases_list:
        if release["tag_name"] == tag:
            for asset in release["assets"]:
                asset_name = bundle_name_pattern.split(asset["name"])[0]
                if asset_name not in bundles_info:
                    bundles_info[asset_name] = {}
                asset_version = asset["name"].split(f"{asset_name}_v")[-1].replace(".zip", "")
                bundles_info[asset_name][asset_version] = {
                    "id": asset["id"],
                    "name": asset["name"],
                    "size": asset["size"],
                    "download_count": asset["size"],
                    "browser_download_url": asset["browser_download_url"],
                }
            return bundles_info
    return bundles_info


def get_bundles_list(
    repo: str = "Project-MONAI/model-zoo",
    tag: str = "hosting_storage_v1",
):
    """
    Get all bundles that are stored in a repository's release that has the provided tag.
    The default values of arguments correspond to the release of MONAI model zoo.

    Args:
        repo: it should be in the form of "repo_owner/repo_name/".
        tag: the tag name of the release.

    Returns:
        a list of bundle names.

    """

    bundle_info = _get_all_bundles_info(repo=repo, tag=tag)

    return list(bundle_info.keys())


def load(
    name: str,
    model_file: Optional[str] = None,
    load_ts_module: bool = False,
    bundle_dir: Optional[PathLike] = None,
    source: str = "github",
    repo: str = "Project-MONAI/model-zoo/hosting_storage_v1",
    progress: bool = True,
    device: Optional[str] = None,
    key_in_ckpt: Optional[str] = None,
    config_files: Sequence[str] = (),
    net_name: Optional[str] = None,
    **net_kwargs,
):
    """
    Load model weights or TorchScript module of a bundle.

    Args:
        name: bundle name.
        model_file: the relative path of the model weights or TorchScript module within bundle.
            If `None`, "models/model.pt" or "models/model.ts" will be used.
        load_ts_module: a flag to specify if loading the TorchScript module.
        bundle_dir: directory the weights/TorchScript module will be loaded from.
            Default is `bundle` subfolder under `torch.hub.get_dir()`.
        source: storage location name. This argument is used when `model_file` is not existing locally and need to be
            downloaded first. "github" is currently the only supported value.
        repo: repo name. This argument is used when `model_file` is not existing locally and need to be
            downloaded first. If `source` is "github", it should be in the form of "repo_owner/repo_name/release_tag".
        progress: whether to display a progress bar when downloading.
        device: target device of returned weights or module, if `None`, prefer to "cuda" if existing.
        key_in_ckpt: for nested checkpoint like `{"model": XXX, "optimizer": XXX, ...}`, specify the key of model
            weights. if not nested checkpoint, no need to set.
        config_files: extra filenames would be loaded. The argument only works when loading a TorchScript module,
            see `_extra_files` in `torch.jit.load` for more details.
        net_name: if not `None`, a corresponding network will be instantiated and load the achieved weights.
            This argument only works when loading weights.
        net_kwargs: other arguments that are used to instantiate the network class defined by `net_name`.

    Returns:
        1. If `load_ts_module` is `False` and `net_name` is `None`, return model weights.
        2. If `load_ts_module` is `False` and `net_name` is not `None`,
            return an instantiated network that loaded the weights.
        3. If `load_ts_module` is `True`, return a triple that include a TorchScript module,
            the corresponding metadata dict, and extra files dict.
            please check `monai.data.load_net_with_metadata` for more details.

    """
    bundle_dir_ = _process_bundle_dir(bundle_dir)

    if model_file is None:
        model_file = os.path.join("models", "model.ts" if load_ts_module is True else "model.pt")
    full_path = os.path.join(bundle_dir_, name, model_file)
    if not os.path.exists(full_path):
        download(name=name, bundle_dir=bundle_dir_, source=source, repo=repo, progress=progress)

    if device is None:
        device = "cuda:0" if is_available() else "cpu"
    # loading with `torch.jit.load`
    if load_ts_module is True:
        return load_net_with_metadata(full_path, map_location=torch.device(device), more_extra_files=config_files)
    # loading with `torch.load`
    model_dict = torch.load(full_path, map_location=torch.device(device))
    if not isinstance(model_dict, Mapping):
        warnings.warn(f"the state dictionary from {full_path} should be a dictionary but got {type(model_dict)}.")
        model_dict = get_state_dict(model_dict)

    if net_name is None:
        return model_dict
    net_kwargs["_target_"] = net_name
    configer = ConfigComponent(config=net_kwargs)
    model = configer.instantiate()
    model.to(device)  # type: ignore
    copy_model_state(dst=model, src=model_dict if key_in_ckpt is None else model_dict[key_in_ckpt])  # type: ignore
    return model


def run(
    runner_id: Optional[Union[str, Sequence[str]]] = None,
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
        python -m monai.bundle run training --meta_file <meta path> --config_file <config path>

        # Override config values at runtime by specifying the component id and its new value:
        python -m monai.bundle run training --net#input_chns 1 ...

        # Override config values with another config file `/path/to/another.json`:
        python -m monai.bundle run evaluating --net %/path/to/another.json ...

        # Override config values with part content of another config file:
        python -m monai.bundle run training --net %/data/other.json#net_arg ...

        # Set default args of `run` in a JSON / YAML file, help to record and simplify the command line.
        # Other args still can override the default args at runtime:
        python -m monai.bundle run --args_file "/workspace/data/args.json" --config_file <config path>

    Args:
        runner_id: ID name of the expected config expression to run, can also be a list of IDs to run in order.
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
    parser.update(pairs=_args)

    # resolve and execute the specified runner expressions in the config, return the results
    return [parser.get_parsed_content(i, lazy=True, eval_expr=True, instantiate=True) for i in ensure_tuple(runner_id_)]


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
    except ValidationError as e:  # pylint: disable=E0712
        # as the error message is very long, only extract the key information
        raise ValueError(
            re.compile(r".*Failed validating", re.S).findall(str(e))[0] + f" against schema `{url}`."
        ) from e
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
        spatial_shape = _get_fake_spatial_shape(input_spatial_shape, p=p_, n=n_, any=any_)
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

        python -m monai.bundle ckpt_export network --filepath <export path> --ckpt_file <checkpoint path> ...

    Args:
        net_id: ID name of the network component in the config, it must be `torch.nn.Module`.
        filepath: filepath to export, if filename has no extension it becomes `.ts`.
        ckpt_file: filepath of the model checkpoint to load.
        meta_file: filepath of the metadata file, if it is a list of file paths, the content of them will be merged.
        config_file: filepath of the config file to save in TorchScript model and extract network information,
            the saved key in the TorchScript model is the config filename without extension, and the saved config
            value is always serialized in JSON format no matter the original file format is JSON or YAML.
            it can be a single file or a list of files. if `None`, must be provided in `args_file`.
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
    _log_input_summary(tag="ckpt_export", args=_args)
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
        ckpt = torch.load(ckpt_file_)
        copy_model_state(dst=net, src=ckpt if key_in_ckpt_ == "" else ckpt[key_in_ckpt_])

    # convert to TorchScript model and save with metadata, config content
    net = convert_to_torchscript(model=net)

    extra_files: Dict = {}
    for i in ensure_tuple(config_file_):
        # split the filename and directory
        filename = os.path.basename(i)
        # remove extension
        filename, _ = os.path.splitext(filename)
        # because all files are stored as JSON their name parts without extension must be unique
        if filename in extra_files:
            raise ValueError(f"Filename part '{filename}' is given multiple times in config file list.")
        # the file may be JSON or YAML but will get loaded and dumped out again as JSON
        extra_files[filename] = json.dumps(ConfigParser.load_config_file(i)).encode()

    # add .json extension to all extra files which are always encoded as JSON
    extra_files = {k + ".json": v for k, v in extra_files.items()}

    save_net_with_metadata(
        jit_obj=net,
        filename_prefix_or_stream=filepath_,
        include_config_vals=False,
        append_timestamp=False,
        meta_values=parser.get().pop("_meta_", None),
        more_extra_files=extra_files,
    )
    logger.info(f"exported to TorchScript file: {filepath_}.")


def init_bundle(
    bundle_dir: PathLike,
    ckpt_file: Optional[PathLike] = None,
    network: Optional[torch.nn.Module] = None,
    metadata_str: Union[Dict, str] = DEFAULT_METADATA,
    inference_str: Union[Dict, str] = DEFAULT_INFERENCE,
):
    """
    Initialise a new bundle directory with some default configuration files and optionally network weights.

    Typical usage example:

    .. code-block:: bash

        python -m monai.bundle init_bundle /path/to/bundle_dir network_ckpt.pt

    Args:
        bundle_dir: directory name to create, must not exist but parent direct must exist
        ckpt_file: optional checkpoint file to copy into bundle
        network: if given instead of ckpt_file this network's weights will be stored in bundle
    """

    bundle_dir = Path(bundle_dir).absolute()

    if bundle_dir.exists():
        raise ValueError(f"Specified bundle directory '{str(bundle_dir)}' already exists")

    if not bundle_dir.parent.is_dir():
        raise ValueError(f"Parent directory of specified bundle directory '{str(bundle_dir)}' does not exist")

    configs_dir = bundle_dir / "configs"
    models_dir = bundle_dir / "models"
    docs_dir = bundle_dir / "docs"

    bundle_dir.mkdir()
    configs_dir.mkdir()
    models_dir.mkdir()
    docs_dir.mkdir()

    if isinstance(metadata_str, dict):
        metadata_str = json.dumps(metadata_str, indent=4)

    if isinstance(inference_str, dict):
        inference_str = json.dumps(inference_str, indent=4)

    with open(str(configs_dir / "metadata.json"), "w") as o:
        o.write(metadata_str)

    with open(str(configs_dir / "inference.json"), "w") as o:
        o.write(inference_str)

    with open(str(docs_dir / "README.md"), "w") as o:
        readme = """
        # Your Model Name

        Describe your model here and how to run it, for example using `inference.json`:

        ```
        python -m monai.bundle run evaluating \
            --meta_file /path/to/bundle/configs/metadata.json \
            --config_file /path/to/bundle/configs/inference.json \
            --dataset_dir ./input \
            --bundle_root /path/to/bundle
        ```
        """

        o.write(dedent(readme))

    with open(str(docs_dir / "license.txt"), "w") as o:
        o.write("Select a license and place its terms here\n")

    if ckpt_file is not None:
        copyfile(str(ckpt_file), str(models_dir / "model.pt"))
    elif network is not None:
        save_state(network, str(models_dir / "model.pt"))

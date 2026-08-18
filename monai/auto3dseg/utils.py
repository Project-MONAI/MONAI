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

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
import warnings
from copy import deepcopy
from numbers import Number
from typing import Any, cast

import numpy as np
import torch

from monai.auto3dseg import Algo
from monai.bundle.config_parser import ConfigParser
from monai.bundle.utils import ID_SEP_KEY
from monai.config import PathLike
from monai.data.meta_tensor import MetaTensor
from monai.transforms import CropForeground, ToCupy
from monai.utils import min_version, optional_import, run_cmd
from monai.utils.deprecate_utils import deprecated
from monai.utils.misc import MONAIEnvVars

__all__ = [
    "get_foreground_image",
    "get_foreground_label",
    "get_label_ccp",
    "concat_val_to_np",
    "concat_multikeys_to_dict",
    "datafold_read",
    "verify_report_format",
    "algo_to_json",
    "algo_from_json",
    "algo_to_pickle",
    "algo_from_pickle",
]

_PICKLE_DISABLED_MSG = (
    "Pickle serialization for Auto3DSeg is disabled by default for security reasons. "
    "Set the environment variable MONAI_ALLOW_PICKLE=1 to enable. Pickle can execute "
    "arbitrary code on load — only enable for files from trusted sources. "
    "Prefer algo_to_json / algo_from_json."
)


def _require_pickle_allowed() -> None:
    if not MONAIEnvVars.allow_pickle():
        raise RuntimeError(_PICKLE_DISABLED_MSG)


measure_np, has_measure = optional_import("skimage.measure", "0.14.2", min_version)
cp, has_cp = optional_import("cupy")


def get_foreground_image(image: MetaTensor) -> np.ndarray:
    """
    Get a foreground image by removing all-zero rectangles on the edges of the image
    Note for the developer: update select_fn if the foreground is defined differently.

    Args:
        image: ndarray image to segment.

    Returns:
        ndarray of foreground image by removing all-zero edges.

    Notes:
        the size of the output is smaller than the input.
    """

    copper = CropForeground(select_fn=lambda x: x > 0, allow_smaller=True)
    image_foreground = copper(image)
    return cast(np.ndarray, image_foreground)


def get_foreground_label(image: MetaTensor, label: MetaTensor) -> MetaTensor:
    """
    Get foreground image pixel values and mask out the non-labeled area.

    Args
        image: ndarray image to segment.
        label: ndarray the image input and annotated with class IDs.

    Returns:
        1D array of foreground image with label > 0
    """

    label_foreground = MetaTensor(image[label > 0])
    return label_foreground


def get_label_ccp(mask_index: MetaTensor, use_gpu: bool = True) -> tuple[list[Any], int]:
    """
    Find all connected components and their bounding shape. Backend can be cuPy/cuCIM or Numpy
    depending on the hardware.

    Args:
        mask_index: a binary mask.
        use_gpu: a switch to use GPU/CUDA or not. If GPU is unavailable, CPU will be used
            regardless of this setting.

    """
    skimage, has_cucim = optional_import("cucim.skimage")
    shape_list = []
    if mask_index.device.type == "cuda" and has_cp and has_cucim and use_gpu:
        mask_cupy = ToCupy()(mask_index.short())
        labeled = skimage.measure.label(mask_cupy)
        vals = cp.unique(labeled[cp.nonzero(labeled)])

        for ncomp in vals:
            comp_idx = cp.argwhere(labeled == ncomp)
            comp_idx_min = cp.min(comp_idx, axis=0).tolist()
            comp_idx_max = cp.max(comp_idx, axis=0).tolist()
            bbox_shape = [comp_idx_max[i] - comp_idx_min[i] + 1 for i in range(len(comp_idx_max))]
            shape_list.append(bbox_shape)
        ncomponents = len(vals)

        del mask_cupy, labeled, vals, comp_idx, ncomp
        cp.get_default_memory_pool().free_all_blocks()

    elif has_measure:
        labeled, ncomponents = measure_np.label(mask_index.data.cpu().numpy(), background=-1, return_num=True)
        for ncomp in range(1, ncomponents + 1):
            comp_idx = np.argwhere(labeled == ncomp)
            comp_idx_min = np.min(comp_idx, axis=0).tolist()
            comp_idx_max = np.max(comp_idx, axis=0).tolist()
            bbox_shape = [comp_idx_max[i] - comp_idx_min[i] + 1 for i in range(len(comp_idx_max))]
            shape_list.append(bbox_shape)
    else:
        raise RuntimeError("Cannot find one of the following required dependencies: {cuPy+cuCIM} or {scikit-image}")

    return shape_list, ncomponents


def concat_val_to_np(
    data_list: list[dict],
    fixed_keys: list[str | int],
    ragged: bool | None = False,
    allow_missing: bool | None = False,
    **kwargs: Any,
) -> np.ndarray:
    """
    Get the nested value in a list of dictionary that shares the same structure.

    Args:
       data_list: a list of dictionary {key1: {key2: np.ndarray}}.
       fixed_keys: a list of keys that records to path to the value in the dict elements.
       ragged: if True, numbers can be in list of lists or ragged format so concat mode needs change.
       allow_missing: if True, it will return a None if the value cannot be found.

    Returns:
        nd.array of concatenated array.

    """

    np_list: list[np.ndarray | None] = []
    for data in data_list:
        parser = ConfigParser(data)
        for i, key in enumerate(fixed_keys):
            fixed_keys[i] = str(key)

        val: Any
        val = parser.get(ID_SEP_KEY.join(fixed_keys))  # type: ignore

        if val is None:
            if allow_missing:
                np_list.append(None)
            else:
                raise AttributeError(f"{fixed_keys} is not nested in the dictionary")
        elif isinstance(val, list):
            np_list.append(np.array(val))
        elif isinstance(val, (torch.Tensor, MetaTensor)):
            np_list.append(val.cpu().numpy())
        elif isinstance(val, np.ndarray):
            np_list.append(val)
        elif isinstance(val, Number):
            np_list.append(np.array(val))
        else:
            raise NotImplementedError(f"{val.__class__} concat is not supported.")

    if allow_missing:
        np_list = [x for x in np_list if x is not None]

    if len(np_list) == 0:
        return np.array([0])
    elif ragged:
        return np.concatenate(np_list, **kwargs)  # type: ignore
    else:
        return np.concatenate([np_list], **kwargs)


def concat_multikeys_to_dict(
    data_list: list[dict], fixed_keys: list[str | int], keys: list[str], zero_insert: bool = True, **kwargs: Any
) -> dict[str, np.ndarray]:
    """
    Get the nested value in a list of dictionary that shares the same structure iteratively on all keys.
    It returns a dictionary with keys with the found values in nd.ndarray.

    Args:
        data_list: a list of dictionary {key1: {key2: np.ndarray}}.
        fixed_keys: a list of keys that records to path to the value in the dict elements.
        keys: a list of string keys that will be iterated to generate a dict output.
        zero_insert: insert a zero in the list so that it can find the value in element 0 before getting the keys
        flatten: if True, numbers are flattened before concat.

    Returns:
        a dict with keys - nd.array of concatenated array pair.
    """

    ret_dict = {}
    for key in keys:
        addon: list[str | int] = [0, key] if zero_insert else [key]
        val = concat_val_to_np(data_list, fixed_keys + addon, **kwargs)
        ret_dict.update({key: val})

    return ret_dict


def datafold_read(datalist: str | dict, basedir: str, fold: int = 0, key: str = "training") -> tuple[list, list]:
    """
    Read a list of data dictionary `datalist`

    Args:
        datalist: the name of a JSON file listing the data, or a dictionary.
        basedir: directory of image files.
        fold: which fold to use (0..1 if in training set).
        key: usually 'training' , but can try 'validation' or 'testing' to get the list data without labels (used in challenges).

    Returns:
        A tuple of two arrays (training, validation).
    """

    if isinstance(datalist, str):
        json_data = ConfigParser.load_config_file(datalist)
    else:
        json_data = datalist

    dict_data = deepcopy(json_data[key])

    for d in dict_data:
        for k, _ in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in dict_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def verify_report_format(report: dict, report_format: dict) -> bool:
    """
    Compares the report and the report_format that has only keys.

    Args:
        report: dict that has real values.
        report_format: dict that only has keys and list-nested value.
    """
    for k_fmt, v_fmt in report_format.items():
        if k_fmt not in report:
            return False

        v = report[k_fmt]

        if isinstance(v_fmt, list) and isinstance(v, list):
            if len(v_fmt) != 1:
                warnings.warn("list length in report_format is not 1", stacklevel=2)
            if len(v_fmt) > 0 and len(v) > 0:
                return verify_report_format(v[0], v_fmt[0])
            else:
                return False

    return True


def _make_json_serializable(value: Any) -> Any:
    """
    Convert a value to a JSON-serializable type.

    Handles numpy arrays, Path objects, torch tensors, and other common types.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    # Fallback to string representation
    return str(value)


def _add_path_with_parent(paths: list[str], path: PathLike | None) -> None:
    """Add a path and its parent directory to the list if the path is a valid directory."""
    if path and os.path.isdir(str(path)):
        abs_path = os.path.abspath(str(path))
        paths.append(abs_path)
        paths.append(os.path.abspath(os.path.join(abs_path, "..")))


def algo_to_json(algo: Algo, template_path: PathLike | None = None, **algo_meta_data: Any) -> str:
    """
    Export the Algo object to a JSON file (pickle-free serialization).

    Args:
        algo: Algo-like object (typically BundleAlgo or subclass).
        template_path: path needed for sys.path setup when loading custom Algo classes.
        algo_meta_data: additional metadata to save (e.g., best_metric, score).

    Returns:
        Filename of the saved Algo object (algo_object.json).
    """
    state = {str(k): _make_json_serializable(v) for k, v in algo.state_dict().items()}

    # Build target string for dynamic class instantiation
    cls = algo.__class__
    target = f"{cls.__module__}.{cls.__name__}"

    algo_meta_data = {str(k): _make_json_serializable(v) for k, v in algo_meta_data.items()}
    data: dict[str, Any] = {
        "_target_": target,
        "_state_": state,
        "template_path": str(template_path) if template_path else (str(getattr(algo, "template_path", "")) or None),
        **algo_meta_data,
    }

    json_filename = os.path.join(algo.get_output_path(), "algo_object.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    return json_filename


def _load_legacy_pickle(pkl_filename: str, template_path: PathLike | None = None) -> Any:
    """
    Load an Algo object from a legacy pickle file.

    This is an internal function to support backward compatibility with pickle files.
    Gated behind ``MONAI_ALLOW_PICKLE=1`` because unpickling executes arbitrary code.
    """
    _require_pickle_allowed()

    with open(pkl_filename, "rb") as f_pi:
        data_bytes = f_pi.read()
    data = pickle.loads(data_bytes)

    if not isinstance(data, dict):
        raise ValueError(f"the data object is {data.__class__}. Dict is expected.")

    if "algo_bytes" not in data:
        raise ValueError(f"key [algo_bytes] not found in {data}. Unable to instantiate.")

    algo_bytes = data.pop("algo_bytes")
    algo_template_path = data.pop("template_path", None)

    template_paths_candidates: list[str] = []
    _add_path_with_parent(template_paths_candidates, template_path)
    _add_path_with_parent(template_paths_candidates, algo_template_path)

    pkl_dir = os.path.dirname(pkl_filename)
    fuzzy_path = os.path.join(pkl_dir, "..", "algorithm_templates")
    if os.path.isdir(fuzzy_path):
        template_paths_candidates.append(os.path.abspath(fuzzy_path))

    if len(template_paths_candidates) == 0:
        algo = pickle.loads(algo_bytes)
        algo.template_path = None
    else:
        for i, p in enumerate(template_paths_candidates):
            path_added = False
            try:
                if p not in sys.path:
                    sys.path.insert(0, p)
                    path_added = True
                algo = pickle.loads(algo_bytes)
                break
            except ModuleNotFoundError as not_found_err:
                logging.debug(f"Folder {p} doesn't contain the Algo templates for Algo instantiation.")
                if i == len(template_paths_candidates) - 1:
                    raise ValueError(
                        f"Failed to instantiate {pkl_filename} with {template_paths_candidates}"
                    ) from not_found_err
            finally:
                if path_added and p in sys.path:
                    sys.path.remove(p)
        algo.template_path = p

    if os.path.abspath(pkl_dir) != os.path.abspath(algo.get_output_path()):
        logging.debug(f"{algo.get_output_path()} is changed. Now override the Algo output_path with: {pkl_dir}.")
        algo.output_path = pkl_dir

    algo_meta_data = dict(data)
    return algo, algo_meta_data


def algo_from_json(filename: str, template_path: PathLike | None = None, **kwargs: Any) -> Any:
    """
    Import the Algo object from a JSON file (pickle-free serialization).

    Args:
        filename: the name of the saved file (algo_object.json or legacy algo_object.pkl).
        template_path: a folder containing files to instantiate the Algo. Besides the `template_path`,
            this function will also attempt to use the `template_path` saved in the file and a directory
            named `algorithm_templates` in the parent folder of the folder containing the file.
        kwargs: additional keyword arguments (reserved for future use).

    Returns:
        algo: the Algo object saved in the file.
        algo_meta_data: additional keyword saved in the file, for example, acc/best_metrics.

    Raises:
        ValueError: if the file format is invalid or the Algo class cannot be instantiated.
        ModuleNotFoundError: if it is unable to instantiate the Algo class.
    """
    abs_filename = os.path.abspath(filename)
    file_dir = os.path.dirname(abs_filename)

    # Check if this is a legacy pickle file
    if filename.endswith(".pkl"):
        warnings.warn(
            "Loading from pickle format (.pkl) is deprecated and will be removed in a future release. "
            "Please re-save your algo using algo_to_json() to convert to the new JSON format.",
            FutureWarning,
            stacklevel=2,
        )
        return _load_legacy_pickle(filename, template_path)

    with open(filename, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"The data object type is {type(data)}, type dict is expected.")

    file_template_path = data.pop("template_path", None)

    if "_target_" not in data:
        raise ValueError(f"Invalid file format: missing '_target_' key in {filename}.")

    target = data.pop("_target_")
    state = data.pop("_state_", {})

    # Build template path candidates for sys.path setup
    template_paths: list[str] = []
    _add_path_with_parent(template_paths, str(template_path) if template_path else None)
    # Handle string "None" from corrupted files
    if file_template_path and file_template_path != "None":
        _add_path_with_parent(template_paths, file_template_path)
    fuzzy_path = os.path.join(file_dir, "..", "algorithm_templates")
    if os.path.isdir(fuzzy_path):
        template_paths.append(os.path.abspath(fuzzy_path))

    # Try each template path to instantiate the class
    paths_to_try: list[str | None] = list(template_paths) if template_paths else [None]
    algo = None
    used_template_path: str | None = None
    last_error: ModuleNotFoundError | None = None
    for path in paths_to_try:
        path_added = False
        try:
            if path and path not in sys.path:
                sys.path.insert(0, path)
                path_added = True

            algo_config: dict[str, Any] = {"_target_": target}
            state_template_path = state.get("template_path")
            if state_template_path:
                algo_config["template_path"] = state_template_path

            parser = ConfigParser(algo_config)
            algo = parser.get_parsed_content()
            used_template_path = path
            break
        except ModuleNotFoundError as e:
            last_error = e
            logging.debug(f"Failed to instantiate {target} with path {path}: {e}")
            continue
        finally:
            if path_added and path in sys.path:
                sys.path.remove(path)

    if algo is None:
        raise ValueError(
            f"Failed to instantiate Algo from target '{target}' with paths {template_paths}"
        ) from last_error

    # Restore the state
    algo.load_state_dict(state)

    # Use the path that successfully imported the class, not the original saved path
    # (the original path may no longer exist if the workdir was moved)
    algo.template_path = used_template_path

    if file_dir != os.path.abspath(algo.get_output_path()):
        logging.debug(f"{algo.get_output_path()} is changed. Now override the Algo output_path with: {file_dir}.")
        algo.output_path = file_dir

    return algo, dict(data)


def list_to_python_fire_arg_str(args: list) -> str:
    """
    Convert a list of arguments to a string that can be used in python-fire.

    Args:
        args: the list of arguments.

    Returns:
        the string that can be used in python-fire.
    """
    args_str = ",".join([str(arg) for arg in args])
    return f"'{args_str}'"


def check_and_set_optional_args(params: dict) -> str:
    """convert `params` into '--key_1=value_1 --key_2=value_2 ...'"""
    cmd_mod_opt = ""
    for k, v in params.items():
        if isinstance(v, dict):
            raise ValueError("Nested dict is not supported.")
        elif isinstance(v, list):
            v = list_to_python_fire_arg_str(v)
        cmd_mod_opt += f" --{k}={v}"
    return cmd_mod_opt


def _prepare_cmd_default(cmd: str, cmd_prefix: str | None = None, **kwargs: Any) -> str:
    """
    Prepare the command for subprocess to run the script with the given arguments.

    Args:
        cmd: the command or script to run in the distributed job.
        cmd_prefix: the command prefix to run the script,
            e.g., "python", "python -m", "python3", "/opt/conda/bin/python3.10".
        kwargs: the keyword arguments to be passed to the script.

    Returns:
        the command to run with ``subprocess``.

    Examples:
        To prepare a subprocess command
        "python train.py run -k --config 'a,b'", the function can be called as
        - _prepare_cmd_default("train.py run -k", config=['a','b'])
        - _prepare_cmd_default("train.py run -k --config 'a,b'")

    """
    params = kwargs.copy()

    if not cmd_prefix or "None" in cmd_prefix:  # defaulting to 'python'
        cmd_prefix = "python"

    if not cmd_prefix.endswith(" "):
        cmd_prefix += " "  # ensure a space after the command prefix so that the script can be appended

    return cmd_prefix + cmd + check_and_set_optional_args(params)


def _prepare_cmd_torchrun(cmd: str, **kwargs: Any) -> str:
    """
    Prepare the command for multi-gpu/multi-node job execution using torchrun.

    Args:
        cmd: the command or script to run in the distributed job.
        kwargs: the keyword arguments to be passed to the script.

    Returns:
        the command to append to ``torchrun``

    Examples:
        For command "torchrun --nnodes=1 --nproc_per_node=8 train.py run -k --config 'a,b'",
        it only prepares command after the torchrun arguments, i.e., "train.py run -k --config 'a,b'".
        The function can be called as
        - _prepare_cmd_torchrun("train.py run -k", config=['a','b'])
        - _prepare_cmd_torchrun("train.py run -k --config 'a,b'")
    """
    params = kwargs.copy()
    return cmd + check_and_set_optional_args(params)


def _prepare_cmd_bcprun(cmd: str, cmd_prefix: str | None = None, **kwargs: Any) -> str:
    """
    Prepare the command for distributed job running using bcprun.

    Args:
        script: the script to run in the distributed job.
        cmd_prefix: the command prefix to run the script, e.g., "python".
        kwargs: the keyword arguments to be passed to the script.

    Returns:
        The command to run the script in the distributed job.

    Examples:
        For command "bcprun -n 2 -p 8 -c python train.py run -k --config 'a,b'",
        it only prepares command after the bcprun arguments, i.e., "train.py run -k --config 'a,b'".
        the function can be called as
        - _prepare_cmd_bcprun("train.py run -k", config=['a','b'], n=2, p=8)
        - _prepare_cmd_bcprun("train.py run -k --config 'a,b'", n=2, p=8)
    """

    return _prepare_cmd_default(cmd, cmd_prefix=cmd_prefix, **kwargs)


def _run_cmd_torchrun(cmd: str, **kwargs: Any) -> subprocess.CompletedProcess:
    """
    Run the command with torchrun.

    Args:
        cmd: the command to run. Typically it is prepared by ``_prepare_cmd_torchrun``.
        kwargs: the keyword arguments to be passed to the ``torchrun``.

    Return:
        the return code of the subprocess command.
    """
    params = kwargs.copy()

    cmd_list = cmd.split()

    # append arguments to the command list
    torchrun_list = ["torchrun"]
    required_args = ["nnodes", "nproc_per_node"]
    for arg in required_args:
        if arg not in params:
            raise ValueError(f"Missing required argument {arg} for torchrun.")
        torchrun_list += [f"--{arg}", str(params.pop(arg))]
    torchrun_list += cmd_list
    return run_cmd(torchrun_list, run_cmd_verbose=True, **params)


def _run_cmd_bcprun(cmd: str, **kwargs: Any) -> subprocess.CompletedProcess:
    """
    Run the command with bcprun.

    Args:
        cmd: the command to run. Typically it is prepared by ``_prepare_cmd_bcprun``.
        kwargs: the keyword arguments to be passed to the ``bcprun``.

    Returns:
        the return code of the subprocess command.
    """
    params = kwargs.copy()
    cmd_list = ["bcprun"]
    required_args = ["n", "p"]
    for arg in required_args:
        if arg not in params:
            raise ValueError(f"Missing required argument {arg} for bcprun.")
        cmd_list += [f"-{arg}", str(params.pop(arg))]
    cmd_list.extend(["-c", cmd])
    return run_cmd(cmd_list, run_cmd_verbose=True, **params)


@deprecated(since="1.6", msg_suffix="Use algo_to_json instead. Pickle is unsafe; see MONAI_ALLOW_PICKLE.")
def algo_to_pickle(algo: Algo, template_path: PathLike | None = None, **algo_meta_data: Any) -> str:
    """Export the Algo object to a pickle file. **Unsafe**; prefer ``algo_to_json``.

    Pickle can execute arbitrary code on load. This function is disabled unless the
    environment variable ``MONAI_ALLOW_PICKLE=1`` is set, and emits a ``UserWarning``
    even when enabled. Use ``algo_to_json`` for safe, pickle-free serialization.

    Args:
        algo: Algo-like object.
        template_path: a str path that is needed to be added to ``sys.path`` to
            instantiate the class on load.
        algo_meta_data: additional keywords to save (e.g., acc/best_metrics).

    Returns:
        Filename of the pickled Algo object.
    """
    _require_pickle_allowed()
    data = {"algo_bytes": pickle.dumps(algo), "template_path": str(template_path)}
    pkl_filename = os.path.join(algo.get_output_path(), "algo_object.pkl")
    for k, v in algo_meta_data.items():
        data.update({k: v})
    data_bytes = pickle.dumps(data)
    with open(pkl_filename, "wb") as f_pi:
        f_pi.write(data_bytes)
    return pkl_filename


@deprecated(since="1.6", msg_suffix="Use algo_from_json instead. Pickle is unsafe; see MONAI_ALLOW_PICKLE.")
def algo_from_pickle(pkl_filename: str, template_path: PathLike | None = None, **kwargs: Any) -> Any:
    """Import the Algo object from a pickle file. **Unsafe**; prefer ``algo_from_json``.

    Disabled unless ``MONAI_ALLOW_PICKLE=1`` is set. See ``_load_legacy_pickle`` for
    template-path resolution details.
    """
    return _load_legacy_pickle(pkl_filename, template_path)

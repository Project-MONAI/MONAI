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

import ast
import importlib
import ntpath
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

import torch

from monai.apps import download_and_extract
from monai.apps.utils import get_logger
from monai.auto3dseg.algo_gen import Algo, AlgoGen
from monai.auto3dseg.utils import (
    _prepare_cmd_bcprun,
    _prepare_cmd_default,
    _prepare_cmd_torchrun,
    _reject_non_algo_target,
    _run_cmd_bcprun,
    _run_cmd_torchrun,
    algo_to_json,
)
from monai.bundle.config_item import ConfigExpression
from monai.bundle.config_parser import ConfigParser
from monai.bundle.reference_resolver import ReferenceResolver
from monai.config import PathLike
from monai.utils import ensure_tuple, look_up_option, run_cmd
from monai.utils.enums import AlgoKeys
from monai.utils.misc import MONAIEnvVars

logger = get_logger(module_name=__name__)
ALGO_HASH = MONAIEnvVars.algo_hash()

__all__ = ["BundleAlgo", "BundleGen"]


def _check_single_path_component(value: Any, what: str) -> None:
    """
    Require ``value`` to be one non-empty directory name with no path separators, drive or NUL character.

    Args:
        value: the name to check.
        what: a description of the value, used in the error message.

    Raises:
        ValueError: if ``value`` is not a plain directory name.
    """
    name = str(value)
    separators = {os.sep, "/"} | ({os.altsep} if os.altsep else set())
    if (
        not name
        or name in (".", "..")
        or "\x00" in name
        or any(sep in name for sep in separators)
        or os.path.isabs(name)
        or ntpath.splitdrive(name)[0]
    ):
        raise ValueError(f"Invalid {what} {value!r}: expected a single directory name without path separators.")


def _confined_join(root: str, name: str) -> str:
    """
    Join ``name`` to ``root`` and check that the resolved result stays inside ``root``.

    Args:
        root: the parent directory.
        name: the child directory name.

    Returns:
        the joined path.

    Raises:
        ValueError: if the resolved path is not inside ``root``.
    """
    path = os.path.join(root, name)
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    if real_path == real_root or os.path.commonpath([real_root, real_path]) != real_root:
        raise ValueError(f"Invalid path: {path!r} is not inside {root!r}.")
    return path


def _resolve_path_value(value: Any, parser: ConfigParser, what: str) -> str:
    """
    Resolve a config value that must be a path string, without evaluating it with ``eval``.

    The config is read from an algorithm output directory, and ``import_bundle_algo_history`` can read one from
    a work directory this process did not write. The value is therefore treated as data, and only these shapes
    are accepted:

    - a plain string, which is returned unchanged;
    - a whole-value reference, ``@name``, where that entry holds a plain string;
    - a ``$`` expression built only from string literals, ``+`` concatenation, ``str()`` with one argument, and
      ``@name`` references to other entries, for example the ``$@bundle_root + '/model_fold' + str(@fold)``
      used by the shipped templates.

    Everything else raises ``ValueError``: any other call, attribute access, a subscript other than the internal
    reference table, a comprehension, a ``_target_`` mapping, and a value that is not a string at all.

    Three consequences are deliberate. A config that relied on a wider expression, for example
    ``$os.path.join(@bundle_root, 'model')`` or an f-string, is now refused, so such a path has to be written out
    in full. References are resolved one level deep only: if ``model_folds_dir`` is itself
    ``$@bundle_root + '/model_folds'`` then ``$@model_folds_dir + '/fold_0'`` is refused, because the referenced
    entry holds an expression rather than a plain value. An integer entry is accepted only inside ``str()``, as
    in ``str(@fold)``, because a path fragment has to be a string.

    Args:
        value: the config value, as stored in the file.
        parser: the parser that holds the other entries referenced with ``@``.
        what: a description of the value, used in the error messages.

    Returns:
        the resolved path string.

    Raises:
        ValueError: if ``value`` is not a string, or is not one of the accepted shapes.
    """
    if not isinstance(value, str):
        raise ValueError(f"Invalid {what}: expected a string, got {type(value).__name__}.")
    if value.startswith("@"):
        resolved = _resolve_path_reference(value[1:], parser, what)
        if not isinstance(resolved, str):
            raise ValueError(f"Invalid {what}: reference {value!r} is not a plain string.")
        return resolved
    if not value.startswith(ConfigExpression.prefix):
        return value

    refs = {
        ref: _resolve_path_reference(ref, parser, what, allow_integer=True)
        for ref in ReferenceResolver.match_refs_pattern(value)
    }
    expression = ReferenceResolver.update_refs_pattern(value, refs)
    try:
        node = ast.parse(expression[len(ConfigExpression.prefix) :], mode="eval").body
    except SyntaxError as e:
        raise ValueError(f"Invalid {what} expression: {value!r}.") from e
    resolved = _path_expression(node, refs, what)
    if not isinstance(resolved, str):
        raise ValueError(f"Invalid {what}: the expression does not resolve to a string.")
    return resolved


def _resolve_path_reference(ref: str, parser: ConfigParser, what: str, allow_integer: bool = False) -> str | int:
    """Return the value of ``@ref``, refusing anything that would need evaluation itself.

    A string is returned as is. A reference whose own value starts with ``$`` or ``@`` is refused, so references
    are never resolved transitively. An integer is only returned when ``allow_integer`` is set, because the
    templates use it through ``str()``, as in ``str(@fold)``; it is never concatenated directly.
    """
    try:
        ref_value = parser[ref]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Invalid {what}: reference @{ref} does not exist.") from e
    if isinstance(ref_value, str):
        if ref_value.startswith(("@", "$")):
            raise ValueError(f"Invalid {what}: reference @{ref} is not a plain value.")
        return ref_value
    if allow_integer and isinstance(ref_value, int) and not isinstance(ref_value, bool):
        return ref_value
    raise ValueError(f"Invalid {what}: reference @{ref} is not a plain value.")


def _path_expression(node: ast.AST, refs: dict[str, str | int], what: str) -> str | int:
    """Evaluate the small expression grammar accepted by :func:`_resolve_path_value`.

    The result is a string for every accepted path fragment. The single exception is a bare reference such as
    ``@fold``, which returns the integer held in the config so that a surrounding ``str(...)`` can format it:
    an expression like ``$@bundle_root + '/model_fold' + str(@fold)`` visits ``@fold`` here and turns it into
    ``'0'``. :func:`_resolve_path_value` rejects a non-string result, so an integer never becomes the path.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _path_expression(node.left, refs, what)
        right = _path_expression(node.right, refs, what)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        raise ValueError(f"Invalid {what}: only strings can be concatenated.")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and len(node.args) == 1
        and not node.keywords
    ):
        return str(_path_expression(node.args[0], refs, what))
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == ReferenceResolver._vars
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
        and node.slice.value in refs
    ):
        return refs[node.slice.value]
    raise ValueError(
        f"Invalid {what}: only string literals, + concatenation, str() of a reference and @references are allowed."
    )


class BundleAlgo(Algo):
    """
    An algorithm represented by a set of bundle configurations and scripts.

    ``BundleAlgo.cfg`` is a ``monai.bundle.ConfigParser`` instance.

    .. code-block:: python

        from monai.apps.auto3dseg import BundleAlgo

        data_stats_yaml = "../datastats.yaml"
        algo = BundleAlgo(template_path="../algorithm_templates")
        algo.set_data_stats(data_stats_yaml)
        # algo.set_data_src("../data_src.json")
        algo.export_to_disk(".", algo_name="segresnet2d_1")

    This class creates MONAI bundles from a directory of 'bundle template'. Different from the regular MONAI bundle
    format, the bundle template may contain placeholders that must be filled using ``fill_template_config`` during
    ``export_to_disk``. Then created bundle keeps the same file structure as the template.

    Note:
        ``get_score`` reads ``ckpt_path`` from ``configs/hyper_parameters.yaml`` in the algorithm output directory
        as data rather than evaluating it, because that directory can come from a work directory this process did
        not write. Only a plain string, a reference to a plain entry, or a ``$`` expression made of string
        literals, ``+``, ``str()`` and ``@references`` are accepted, as in
        ``$@bundle_root + '/model_fold' + str(@fold)``. A saved config that needs anything else, for example a
        call to ``os.path.join`` or a reference to another expression, raises ``ValueError`` instead of being
        evaluated, so a score written by a template that builds its path that way cannot be read back.

    """

    # Keys carried by ``state_dict``/``load_state_dict``. Declared once so that a partially initialized instance
    # (for example one built without running ``__init__``) can still validate the keys of a restored state.
    _STATE_KEYS = (
        "template_path",
        "data_stats_files",
        "data_list_file",
        "mlflow_tracking_uri",
        "mlflow_experiment_name",
        "output_path",
        "name",
        "best_metric",
        "fill_records",
        "device_setting",
    )

    def __init__(self, template_path: PathLike | None = None):
        """
        Create an Algo instance based on the predefined Algo template.

        Args:
            template_path: path to a folder that contains the algorithm templates. If this is not provided, it's value
                must be loaded with `load_state_dict`.
                Please check https://github.com/Project-MONAI/research-contributions/tree/main/auto3dseg/algorithm_templates

        """

        self.template_path = template_path
        self.data_stats_files = ""
        self.data_list_file = ""
        self.mlflow_tracking_uri: str | None = None
        self.mlflow_experiment_name: str | None = None
        self.output_path = ""
        self.name = ""
        self.best_metric = None
        # track records when filling template config: {"<config name>": {"<placeholder key>": value, ...}, ...}
        self.fill_records: dict = {}
        # device_setting set default value and sanity check, in case device_setting not from autorunner
        self.device_setting: dict[str, int | str] = {
            "CUDA_VISIBLE_DEVICES": ",".join([str(x) for x in range(torch.cuda.device_count())]),
            "n_devices": int(torch.cuda.device_count()),
            "NUM_NODES": int(os.environ.get("NUM_NODES", 1)),
            "MN_START_METHOD": os.environ.get("MN_START_METHOD", "bcprun"),
            "CMD_PREFIX": os.environ.get("CMD_PREFIX", ""),
        }

    def pre_check_skip_algo(self, skip_bundlegen: bool = False, skip_info: str = "") -> tuple[bool, str]:
        """
        Analyse the data analysis report and check if the algorithm needs to be skipped.
        This function is overriden within algo.
        Args:
            skip_bundlegen: skip generating bundles for this algo if true.
            skip_info: info to print when skipped.
        """
        return skip_bundlegen, skip_info

    def set_data_stats(self, data_stats_files: str) -> None:
        """
        Set the data analysis report (generated by DataAnalyzer).

        Args:
            data_stats_files: path to the datastats yaml file
        """
        self.data_stats_files = data_stats_files

    def set_data_source(self, data_src_cfg: str) -> None:
        """
        Set the data source configuration file

        Args:
            data_src_cfg: path to a configuration file (yaml) that contains datalist, dataroot, and other params.
                The config will be in a form of {"modality": "ct", "datalist": "path_to_json_datalist", "dataroot":
                "path_dir_data"}
        """
        self.data_list_file = data_src_cfg

    def set_mlflow_tracking_uri(self, mlflow_tracking_uri: str | None) -> None:
        """
        Set the tracking URI for MLflow server

        Args:
            mlflow_tracking_uri: a tracking URI for MLflow server which could be local directory or address of
                the remote tracking Server; MLflow runs will be recorded locally in algorithms' model folder if
                the value is None.
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri

    def set_mlflow_experiment_name(self, mlflow_experiment_name: str | None) -> None:
        """
        Set the experiment name for MLflow server

        Args:
            mlflow_experiment_name: a string to specify the experiment name for MLflow server.
        """
        self.mlflow_experiment_name = mlflow_experiment_name

    def fill_template_config(self, data_stats_filename: str, algo_path: str, **kwargs: Any) -> dict:
        """
        The configuration files defined when constructing this Algo instance might not have a complete training
        and validation pipelines. Some configuration components and hyperparameters of the pipelines depend on the
        training data and other factors. This API is provided to allow the creation of fully functioning config files.
        Return the records of filling template config: {"<config name>": {"<placeholder key>": value, ...}, ...}.

        Args:
            data_stats_filename: filename of the data stats report (generated by DataAnalyzer)

        Notes:
            Template filling is optional. The user can construct a set of pre-filled configs without replacing values
            by using the data analysis results. It is also intended to be re-implemented in subclasses of BundleAlgo
            if the user wants their own way of auto-configured template filling.
        """
        return {}

    def export_to_disk(self, output_path: str, algo_name: str, **kwargs: Any) -> None:
        """
        Fill the configuration templates, write the bundle (configs + scripts) to folder `output_path/algo_name`.

        Args:
            output_path: Path to export the 'scripts' and 'configs' directories.
            algo_name: the identifier of the algorithm (usually contains the name and extra info like fold ID).
                It must be a single directory name: path separators, ``.`` and ``..`` are rejected.
            kwargs: other parameters, including: "copy_dirs=True/False" means whether to copy the template as output
                instead of inplace operation, "fill_template=True/False" means whether to fill the placeholders
                in the template. other parameters are for `fill_template_config` function.


        Raises:
            ValueError: if ``algo_name`` or the Algo ``name`` is not a single directory name, or the destination
                does not resolve inside ``output_path``.
        """
        if kwargs.pop("copy_dirs", True):
            _check_single_path_component(algo_name, "algo_name")
            if self.name:  # an empty name copies the whole template_path
                _check_single_path_component(self.name, "the Algo name")
            source = os.path.join(str(self.template_path), self.name)
            if not os.path.isdir(source):
                raise ValueError(f"Algo template folder {source!r} does not exist.")
            self.output_path = _confined_join(output_path, algo_name)
            os.makedirs(self.output_path, exist_ok=True)
            if os.path.isdir(self.output_path):
                shutil.rmtree(self.output_path)
            # copy algorithm_templates/<Algo> to the working directory output_path
            shutil.copytree(source, self.output_path)
        else:
            self.output_path = str(self.template_path)
        if kwargs.pop("fill_template", True):
            self.fill_records = self.fill_template_config(self.data_stats_files, self.output_path, **kwargs)
        logger.info(f"Generated:{self.output_path}")

    def _create_cmd(self, train_params: None | dict = None) -> tuple[str, str]:
        """
        Create the command to execute training.

        """
        if train_params is None:
            train_params = {}
        params = deepcopy(train_params)

        train_py = os.path.join(self.output_path, "scripts", "train.py")
        config_dir = os.path.join(self.output_path, "configs")

        config_files = []
        if os.path.isdir(config_dir):
            for file in sorted(os.listdir(config_dir)):
                if file.endswith(("yaml", "json")):
                    # Python Fire may be confused by single-quoted WindowsPath
                    config_files.append(Path(os.path.join(config_dir, file)).as_posix())

        if int(self.device_setting["NUM_NODES"]) > 1:
            # multi-node command
            # only bcprun is supported for now
            try:
                look_up_option(self.device_setting["MN_START_METHOD"], ["bcprun"])
            except ValueError as err:
                raise NotImplementedError(
                    f"{self.device_setting['MN_START_METHOD']} is not supported yet."
                    "Try modify BundleAlgo._create_cmd for your cluster."
                ) from err

            return (
                _prepare_cmd_bcprun(
                    f"{shlex.quote(train_py)} run",
                    cmd_prefix=f"{self.device_setting['CMD_PREFIX']}",
                    config_file=config_files,
                    **params,
                ),
                "",
            )
        elif int(self.device_setting["n_devices"]) > 1:
            return _prepare_cmd_torchrun(f"{shlex.quote(train_py)} run", config_file=config_files, **params), ""
        else:
            return (
                _prepare_cmd_default(
                    f"{shlex.quote(train_py)} run",
                    cmd_prefix=f"{self.device_setting['CMD_PREFIX']}",
                    config_file=config_files,
                    **params,
                ),
                "",
            )

    def _run_cmd(self, cmd: str, devices_info: str = "") -> subprocess.CompletedProcess:
        """
        Execute the training command with target devices information.

        """
        if devices_info:
            warnings.warn(f"input devices_info {devices_info} is deprecated and ignored.")

        ps_environ = os.environ.copy()
        ps_environ["CUDA_VISIBLE_DEVICES"] = str(self.device_setting["CUDA_VISIBLE_DEVICES"])

        # delete pattern "VAR=VALUE" at the beginning of the string, with optional leading/trailing whitespaces
        cmd = re.sub(r"^\s*\w+=.*?\s+", "", cmd)

        if int(self.device_setting["NUM_NODES"]) > 1:
            try:
                look_up_option(self.device_setting["MN_START_METHOD"], ["bcprun"])
            except ValueError as err:
                raise NotImplementedError(
                    f"{self.device_setting['MN_START_METHOD']} is not supported yet. Try modify BundleAlgo._run_cmd for your cluster."
                ) from err

            return _run_cmd_bcprun(cmd, n=self.device_setting["NUM_NODES"], p=self.device_setting["n_devices"])
        elif int(self.device_setting["n_devices"]) > 1:
            return _run_cmd_torchrun(
                cmd, nnodes=1, nproc_per_node=self.device_setting["n_devices"], env=ps_environ, check=True
            )
        else:
            return run_cmd(shlex.split(cmd), run_cmd_verbose=True, env=ps_environ, check=True)

    def train(
        self, train_params: None | dict = None, device_setting: None | dict = None
    ) -> subprocess.CompletedProcess:
        """
        Load the run function in the training script of each model. Training parameter is predefined by the
        algo_config.yaml file, which is pre-filled by the fill_template_config function in the same instance.

        Args:
            train_params:  training parameters
            device_setting: device related settings, should follow the device_setting in auto_runner.set_device_info.
                'CUDA_VISIBLE_DEVICES' should be a string e.g. '0,1,2,3'
        """
        if device_setting is not None:
            self.device_setting.update(device_setting)
            self.device_setting["n_devices"] = len(str(self.device_setting["CUDA_VISIBLE_DEVICES"]).split(","))

        if train_params is not None and "CUDA_VISIBLE_DEVICES" in train_params:
            warnings.warn("CUDA_VISIBLE_DEVICES is deprecated from train_params!")
            train_params.pop("CUDA_VISIBLE_DEVICES")

        cmd, _unused_return = self._create_cmd(train_params)
        return self._run_cmd(cmd)

    def get_score(self, *args, **kwargs):
        """
        Returns validation scores of the model trained by the current Algo.

        The configuration is read from the algorithm output directory, which may belong to a work directory this
        process did not create (``import_bundle_algo_history`` reads any work dir), so ``ckpt_path`` is resolved
        as data rather than evaluated. See the class docstring for the syntax that is accepted, which is a
        narrower set than ``eval`` would allow.

        Args:
            *args: unused, kept for compatibility with the ``Algo`` interface.
            **kwargs: unused, kept for compatibility with the ``Algo`` interface.

        Returns:
            the best average Dice score recorded in ``progress.yaml``.

        Raises:
            ValueError: if ``ckpt_path`` in the saved configuration is not a plain path string.
        """
        config_yaml = os.path.join(self.output_path, "configs", "hyper_parameters.yaml")
        parser = ConfigParser(globals=False)
        parser.read_config(config_yaml)
        # resolved as data, never with eval(): the file can come from a work directory this process
        # did not write, and a `$` expression there must not be able to call or import anything
        ckpt_path = _resolve_path_value(parser.get("ckpt_path", default=self.output_path), parser, "ckpt_path")

        dict_file = ConfigParser.load_config_file(os.path.join(ckpt_path, "progress.yaml"))
        # dict_file: a list of scores saved in the form of dict in progress.yaml
        return dict_file[-1]["best_avg_dice_score"]  # the last one is the best one

    def get_inferer(self, *args, **kwargs):
        """
        Load the InferClass from the infer.py. The InferClass should be defined in the template under the path of
        `"scripts/infer.py"`. It is required to define the "InferClass" (name is fixed) with two functions at least
        (``__init__`` and ``infer``). The init class has an override kwargs that can be used to override parameters in
        the run-time optionally.

        Examples:

        .. code-block:: python

            class InferClass
                def __init__(self, config_file: Optional[Union[str, Sequence[str]]] = None, **override):
                    # read configs from config_file (sequence)
                    # set up transforms
                    # set up model
                    # set up other hyper parameters
                    return

                @torch.no_grad()
                def infer(self, image_file):
                    # infer the model and save the results to output
                    return output

        """
        infer_py = os.path.join(self.output_path, "scripts", "infer.py")
        if not os.path.isfile(infer_py):
            raise ValueError(f"{infer_py} is not found, please check the path.")

        config_dir = os.path.join(self.output_path, "configs")
        configs_path = [os.path.join(config_dir, f) for f in os.listdir(config_dir)]

        spec = importlib.util.spec_from_file_location("InferClass", infer_py)
        infer_class = importlib.util.module_from_spec(spec)  # type: ignore
        sys.modules["InferClass"] = infer_class
        spec.loader.exec_module(infer_class)  # type: ignore
        return infer_class.InferClass(configs_path, *args, **kwargs)

    def predict(self, predict_files: list, predict_params: dict | None = None) -> list:
        """
        Use the trained model to predict the outputs with a given input image.

        Args:
            predict_files: a list of paths to files to run inference on ["path_to_image_1", "path_to_image_2"]
            predict_params: a dict to override the parameters in the bundle config (including the files to predict).

        """
        params = {} if predict_params is None else deepcopy(predict_params)
        inferer = self.get_inferer(**params)
        return [inferer.infer(f) for f in ensure_tuple(predict_files)]

    def get_output_path(self):
        """Returns the algo output paths to find the algo scripts and configs."""
        return self.output_path

    def state_dict(self) -> dict:
        """
        Return state for serialization.

        Returns:
            A dictionary containing the BundleAlgo state to serialize.

        Note:
            template_path is excluded as it is determined dynamically at load time
            based on which path successfully imports the Algo class.
        """
        return {key: getattr(self, key) for key in self._STATE_KEYS}

    def load_state_dict(self, state: dict) -> None:
        """
        Restore state from a dictionary.

        Args:
            state: A dictionary containing the state to restore.

        Raises:
            ValueError: if the state has unknown keys, or a ``device_setting`` that is malformed or that asks for
                more devices than the restored ``CUDA_VISIBLE_DEVICES`` exposes.
        """
        allowed = set(self._STATE_KEYS)
        unknown = sorted(set(state) - allowed)
        if unknown:
            raise ValueError(
                f"Unexpected keys in {type(self).__name__} state: {unknown}. Expected a subset of {sorted(allowed)}."
            )
        if "device_setting" in state:
            state = dict(state)
            restored = _validate_device_setting(state["device_setting"])
            # merge into the current settings so fields missing from the saved state keep their defaults
            merged = {**getattr(self, "device_setting", {}), **restored}
            if "CUDA_VISIBLE_DEVICES" in restored:
                visible = _count_visible_devices(restored["CUDA_VISIBLE_DEVICES"])
                if "n_devices" in restored:
                    if merged["n_devices"] > visible:
                        raise ValueError(
                            f'Value in `state["device_setting"]` asks for {merged["n_devices"]} devices but '
                            f"CUDA_VISIBLE_DEVICES={restored['CUDA_VISIBLE_DEVICES']!r} exposes {visible}."
                        )
                else:
                    # a state without n_devices must not keep the device count of this host
                    merged["n_devices"] = visible
            state["device_setting"] = merged
        for key, value in state.items():
            setattr(self, key, value)


def _count_visible_devices(visible: str) -> int:
    """Number of devices selected by a ``CUDA_VISIBLE_DEVICES`` value (a single MIG identifier counts as one)."""
    return len([item for item in str(visible).split(",") if item.strip()])


def _validate_device_setting(device_setting: Any) -> dict:
    """
    Check a restored ``device_setting`` dictionary before it is used to build training commands.

    Args:
        device_setting: the value restored from a saved state.

    Returns:
        a copy of ``device_setting`` with the numeric fields converted to ``int``.

    Raises:
        ValueError: if the value is not a dictionary, has unknown keys, or has malformed fields.
    """
    known = {"CUDA_VISIBLE_DEVICES", "n_devices", "NUM_NODES", "MN_START_METHOD", "CMD_PREFIX"}
    if not isinstance(device_setting, dict):
        raise ValueError(f"device_setting must be a dict, got {type(device_setting).__name__}.")
    unknown = sorted(set(device_setting) - known)
    if unknown:
        raise ValueError(f"Unexpected keys in device_setting: {unknown}.")
    checked = dict(device_setting)
    for key in ("n_devices", "NUM_NODES"):
        if key in checked:
            value = checked[key]
            if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
                raise ValueError(f"device_setting[{key!r}] must be a non-negative integer, got {value!r}.")
            checked[key] = int(value)
    if "CUDA_VISIBLE_DEVICES" in checked and not re.fullmatch(r"[\w,./-]*", str(checked["CUDA_VISIBLE_DEVICES"])):
        # device indices, GPU UUIDs and MIG identifiers (``MIG-GPU-<uuid>/<gi>/<ci>``) are all accepted
        raise ValueError(f"device_setting['CUDA_VISIBLE_DEVICES'] is malformed: {checked['CUDA_VISIBLE_DEVICES']!r}.")
    for key in ("MN_START_METHOD", "CMD_PREFIX"):
        if key in checked and not isinstance(checked[key], str):
            raise ValueError(f"device_setting[{key!r}] must be a string, got {type(checked[key]).__name__}.")
    return checked


# path to download the algo_templates
default_algo_zip = (
    f"https://github.com/Project-MONAI/research-contributions/releases/download/algo_templates/{ALGO_HASH}.tar.gz"
)

# default algorithms
default_algos = {
    "segresnet2d": dict(_target_="segresnet2d.scripts.algo.Segresnet2dAlgo"),
    "dints": dict(_target_="dints.scripts.algo.DintsAlgo"),
    "swinunetr": dict(_target_="swinunetr.scripts.algo.SwinunetrAlgo"),
    "segresnet": dict(_target_="segresnet.scripts.algo.SegresnetAlgo"),
}


def _download_algos_url(url: str, at_path: str) -> dict[str, dict[str, str]]:
    """
    Downloads the algorithm templates release archive, and extracts it into a parent directory of the at_path folder.
    Returns a dictionary of the algorithm templates.
    """
    at_path = os.path.abspath(at_path)
    zip_download_dir = TemporaryDirectory()
    algo_compressed_file = os.path.join(zip_download_dir.name, "algo_templates.tar.gz")

    download_attempts = 3
    for i in range(download_attempts):
        try:
            download_and_extract(url=url, filepath=algo_compressed_file, output_dir=os.path.dirname(at_path))
        except Exception as e:
            msg = f"Download and extract of {url} failed, attempt {i + 1}/{download_attempts}."
            if i < download_attempts - 1:
                warnings.warn(msg)
                time.sleep(i)
            else:
                zip_download_dir.cleanup()
                raise ValueError(msg) from e
        else:
            break

    zip_download_dir.cleanup()

    algos_all = deepcopy(default_algos)
    for name in algos_all:
        algos_all[name]["template_path"] = at_path

    return algos_all


def _copy_algos_folder(folder, at_path):
    """
    Copies the algorithm templates folder to at_path.
    Returns a dictionary of algorithm templates.
    """
    folder = os.path.abspath(folder)
    at_path = os.path.abspath(at_path)

    if folder != at_path:
        if os.path.exists(at_path):
            shutil.rmtree(at_path)
        shutil.copytree(folder, at_path)

    algos_all = {}
    for name in os.listdir(at_path):
        if os.path.exists(os.path.join(folder, name, "scripts", "algo.py")):
            algos_all[name] = dict(_target_=f"{name}.scripts.algo.{name.capitalize()}Algo", template_path=at_path)
            logger.info(f"Copying template: {name} -- {algos_all[name]}")
    if not algos_all:
        raise ValueError(f"Unable to find any algos in {folder}")

    return algos_all


def _algo_from_params(algo_name: Any, algo_params: Any) -> Algo:
    """Instantiate one algorithm from an entry of the ``algos`` mapping, using literal values only.

    The mapping is user input: ``AutoRunner`` reads it from ``input.yaml`` and ``BundleGen``/its CLI accept it
    directly. Its values are therefore plain data, and they are handed to the ``Algo`` subclass as literals.
    They are never passed through ``ConfigParser``, which would treat strings starting with ``$``, ``@`` or
    ``%`` as configuration directives and evaluate them.

    The class named by ``_target_`` is still imported, from ``template_path`` if the module is not installed,
    so ``template_path`` must point to a directory you trust: importing a module runs its top-level code.
    The directory is validated and added to ``sys.path`` here, and left there afterwards, because the
    instantiated ``Algo`` may import further modules from its template when it is trained or exported.

    Args:
        algo_name: the key of the entry, used in error messages.
        algo_params: the mapping with a ``_target_`` key naming an ``Algo`` subclass and optional keyword
            arguments for its constructor, such as ``template_path``.

    Returns:
        the instantiated ``Algo``.

    Raises:
        ValueError: if the entry is not a mapping, has no ``_target_``, the target is not an ``Algo``
            subclass, or ``template_path`` is not a directory.
    """
    if not isinstance(algo_params, dict):
        raise ValueError(
            f"Invalid algos entry {algo_name!r}: expected a dict with a '_target_' key, "
            f"got {type(algo_params).__name__}."
        )

    params = dict(algo_params)
    target = params.pop("_target_", None)
    if target is None:
        raise ValueError(f"Invalid algos entry {algo_name!r}: missing '_target_' key.")

    template_path = params.get("template_path", ".")
    if not isinstance(template_path, str):
        raise ValueError(
            f"Invalid template_path {template_path!r} for algos entry {algo_name!r}: expected a string, "
            f"got {type(template_path).__name__}."
        )
    # check the directory here: a bad sys.path entry only surfaces later, as a ModuleNotFoundError for the
    # target class that says nothing about the path being wrong
    if template_path and not os.path.isdir(template_path):
        raise ValueError(f"Invalid template_path {template_path!r} for algos entry {algo_name!r}: not a directory.")
    if template_path and template_path not in sys.path:
        sys.path.append(template_path)

    # the allowlist resolves the class and checks issubclass(..., Algo) before anything is instantiated
    algo_cls: Any = _reject_non_algo_target(target, f"algos entry {algo_name!r}")
    return algo_cls(**params)


class BundleGen(AlgoGen):
    """
    This class generates a set of bundles according to the cross-validation folds, each of them can run independently.

    Args:
        algo_path: the directory path to save the algorithm templates. Default is the current working dir.
        algos: If dictionary, it outlines the algorithm to use. The values are treated as plain data: each entry
            needs a ``_target_`` key naming an ``Algo`` subclass (a full module path or a monai component name)
            and may pass further keyword arguments for its constructor (for example ``template_path``).
            Configuration expressions (``$``, ``@``, ``%``) are not evaluated. The target class is imported
            from ``template_path`` when the module is not installed, so that directory must be trusted.
            If a list or a string, defines a subset of names of
            the algorithms to use, e.g. ('segresnet', 'dints') out of the full set of algorithm templates provided
            by templates_path_or_url. Defaults to None - to use all available algorithms.
        templates_path_or_url: the folder with the algorithm templates or a url. If None provided, the default template
            zip url will be downloaded and extracted into the algo_path. The current default options are released at:
            https://github.com/Project-MONAI/research-contributions/tree/main/auto3dseg.
        data_stats_filename: the path to the data stats file (generated by DataAnalyzer).
        data_src_cfg_name: the path to the data source config YAML file. The config will be in a form of
                           {"modality": "ct", "datalist": "path_to_json_datalist", "dataroot": "path_dir_data"}.
        mlflow_tracking_uri: a tracking URI for MLflow server which could be local directory or address of
            the remote tracking Server; MLflow runs will be recorded locally in algorithms' model folder if
            the value is None.
        mlflow_experiment_name: a string to specify the experiment name for MLflow server.
    .. code-block:: bash

        python -m monai.apps.auto3dseg BundleGen generate --data_stats_filename="../algorithms/datastats.yaml"
    """

    def __init__(
        self,
        algo_path: str = ".",
        algos: dict | list | str | None = None,
        templates_path_or_url: str | None = None,
        data_stats_filename: str | None = None,
        data_src_cfg_name: str | None = None,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str | None = None,
    ):
        if algos is None or isinstance(algos, (list, tuple, str)):
            if templates_path_or_url is None:
                templates_path_or_url = default_algo_zip

            at_path = os.path.join(os.path.abspath(algo_path), "algorithm_templates")

            if os.path.isdir(templates_path_or_url):
                # if a local folder, copy if necessary
                logger.info(f"BundleGen from directory {templates_path_or_url}")
                algos_all = _copy_algos_folder(folder=templates_path_or_url, at_path=at_path)
            elif urlparse(templates_path_or_url).scheme in ("http", "https"):
                # if url, trigger the download and extract process
                logger.info(f"BundleGen from {templates_path_or_url}")
                algos_all = _download_algos_url(url=templates_path_or_url, at_path=at_path)
            else:
                raise ValueError(f"{self.__class__} received invalid templates_path_or_url: {templates_path_or_url}")

            if algos is not None:
                algos = {k: v for k, v in algos_all.items() if k in ensure_tuple(algos)}  # keep only provided
                if len(algos) == 0:
                    raise ValueError(f"Unable to find provided algos in {algos_all}")
            else:
                algos = algos_all

        self.algos: Any = []
        if isinstance(algos, dict):
            for algo_name, algo_params in sorted(algos.items()):
                try:
                    onealgo: Any = _algo_from_params(algo_name, algo_params)
                    onealgo.name = algo_name
                    self.algos.append(onealgo)
                except RuntimeError as e:
                    msg = """Please make sure the folder structure of an Algo Template follows
                        [algo_name]
                        ├── configs
                        │   ├── hyper_parameters.yaml  # automatically generated yaml from a set of ``template_configs``
                        └── scripts
                            ├── test.py
                            ├── __init__.py
                            └── validate.py
                    """
                    raise RuntimeError(msg) from e
        else:
            raise ValueError("Unexpected error algos is not a dict")

        self.data_stats_filename = data_stats_filename
        self.data_src_cfg_name = data_src_cfg_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.history: list[dict] = []

    def set_data_stats(self, data_stats_filename: str) -> None:
        """
        Set the data stats filename

        Args:
            data_stats_filename: filename of datastats
        """
        self.data_stats_filename = data_stats_filename

    def get_data_stats(self):
        """Get the filename of the data stats"""
        return self.data_stats_filename

    def set_data_src(self, data_src_cfg_name):
        """
        Set the data source filename

        Args:
            data_src_cfg_name: filename of data_source file
        """
        self.data_src_cfg_name = data_src_cfg_name

    def get_data_src(self):
        """Get the data source filename"""
        return self.data_src_cfg_name

    def set_mlflow_tracking_uri(self, mlflow_tracking_uri):
        """
        Set the tracking URI for MLflow server

        Args:
            mlflow_tracking_uri: a tracking URI for MLflow server which could be local directory or address of
                the remote tracking Server; MLflow runs will be recorded locally in algorithms' model folder if
                the value is None.
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri

    def set_mlflow_experiment_name(self, mlflow_experiment_name):
        """
        Set the experiment name for MLflow server

        Args:
            mlflow_experiment_name: a string to specify the experiment name for MLflow server.
        """
        self.mlflow_experiment_name = mlflow_experiment_name

    def get_mlflow_tracking_uri(self):
        """Get the tracking URI for MLflow server"""
        return self.mlflow_tracking_uri

    def get_mlflow_experiment_name(self):
        """Get the experiment name for MLflow server"""
        return self.mlflow_experiment_name

    def get_history(self) -> list:
        """Get the history of the bundleAlgo object with their names/identifiers"""
        return self.history

    def generate(
        self,
        output_folder: str = ".",
        num_fold: int = 5,
        gpu_customization: bool = False,
        gpu_customization_specs: dict[str, Any] | None = None,
        allow_skip: bool = True,
    ) -> None:
        """
        Generate the bundle scripts/configs for each bundleAlgo

        Args:
            output_folder: the output folder to save each algorithm.
            num_fold: the number of cross validation fold.
            gpu_customization: the switch to determine automatically customize/optimize bundle script/config
                parameters for each bundleAlgo based on gpus. Custom parameters are obtained through dummy
                training to simulate the actual model training process and hyperparameter optimization (HPO)
                experiments.
            gpu_customization_specs: the dictionary to enable users overwrite the HPO settings. user can
                overwrite part of variables as follows or all of them. The structure is as follows.
            allow_skip: a switch to determine if some Algo in the default templates can be skipped based on the
                analysis on the dataset from Auto3DSeg DataAnalyzer.

                .. code-block:: python

                    gpu_customization_specs = {
                        'ALGO': {
                            'num_trials': 6,
                            'range_num_images_per_batch': [1, 20],
                            'range_num_sw_batch_size': [1, 20]
                        }
                    }

            ALGO: the name of algorithm. It could be one of algorithm names (e.g., 'dints') or 'universal' which
                would apply changes to all algorithms. Possible options are

                - {``"universal"``, ``"dints"``, ``"segresnet"``, ``"segresnet2d"``, ``"swinunetr"``}.

            num_trials: the number of HPO trials/experiments to run.
            range_num_images_per_batch: the range of number of images per mini-batch.
            range_num_sw_batch_size: the range of batch size in sliding-window inferer.
        """
        fold_idx = list(range(num_fold))
        for algo in self.algos:
            for f_id in ensure_tuple(fold_idx):
                data_stats = self.get_data_stats()
                data_src_cfg = self.get_data_src()
                mlflow_tracking_uri = self.get_mlflow_tracking_uri()
                mlflow_experiment_name = self.get_mlflow_experiment_name()
                gen_algo = deepcopy(algo)
                gen_algo.set_data_stats(data_stats)
                gen_algo.set_data_source(data_src_cfg)
                gen_algo.set_mlflow_tracking_uri(mlflow_tracking_uri)
                gen_algo.set_mlflow_experiment_name(mlflow_experiment_name)
                name = f"{gen_algo.name}_{f_id}"

                if allow_skip:
                    skip_bundlegen, skip_info = gen_algo.pre_check_skip_algo()
                    if skip_bundlegen:
                        logger.info(f"{name} is skipped! {skip_info}")
                        continue

                if gpu_customization:
                    gen_algo.export_to_disk(
                        output_folder,
                        name,
                        fold=f_id,
                        gpu_customization=True,
                        gpu_customization_specs=gpu_customization_specs,
                    )
                else:
                    gen_algo.export_to_disk(output_folder, name, fold=f_id)

                algo_to_json(gen_algo, template_path=algo.template_path)
                self.history.append(
                    {AlgoKeys.ID: name, AlgoKeys.ALGO: gen_algo}
                )  # track the previous, may create a persistent history

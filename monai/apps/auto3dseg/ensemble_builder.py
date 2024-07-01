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

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, cast
from warnings import warn

import numpy as np
import torch
import torch.distributed as dist

from monai.apps.auto3dseg.bundle_gen import BundleAlgo
from monai.apps.auto3dseg.utils import get_name_from_algo_id, import_bundle_algo_history
from monai.apps.utils import get_logger
from monai.auto3dseg import concat_val_to_np
from monai.auto3dseg.utils import (
    _prepare_cmd_bcprun,
    _prepare_cmd_torchrun,
    _run_cmd_bcprun,
    _run_cmd_torchrun,
    datafold_read,
)
from monai.bundle import ConfigParser
from monai.data import partition_dataset
from monai.transforms import MeanEnsemble, SaveImage, VoteEnsemble
from monai.utils import RankFilter
from monai.utils.enums import AlgoKeys
from monai.utils.misc import check_kwargs_exist_in_class_init, prob2class
from monai.utils.module import look_up_option, optional_import

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

logger = get_logger(module_name=__name__)


class AlgoEnsemble(ABC):
    """
    The base class of Ensemble methods
    """

    def __init__(self):
        self.algos = []
        self.mode = "mean"
        self.infer_files = []
        self.algo_ensemble = []

    def set_algos(self, infer_algos):
        """
        Register model in the ensemble
        """
        self.algos = deepcopy(infer_algos)

    def get_algo(self, identifier):
        """
        Get a model by identifier.

        Args:
            identifier: the name of the bundleAlgo
        """
        for algo in self.algos:
            if identifier == algo[AlgoKeys.ID]:
                return algo

    def get_algo_ensemble(self):
        """
        Get the algo ensemble after ranking or a empty list if ranking was not started.

        Returns:
            A list of Algo
        """
        return self.algo_ensemble

    def set_infer_files(self, dataroot: str, data_list_or_path: str | list, data_key: str = "testing") -> None:
        """
        Set the files to perform model inference.

        Args:
            dataroot: the path of the files
            data_list_or_path: the data source file path
        """

        self.infer_files = []

        if isinstance(data_list_or_path, list):
            self.infer_files = data_list_or_path
        elif isinstance(data_list_or_path, str):
            datalist = ConfigParser.load_config_file(data_list_or_path)
            if data_key in datalist:
                self.infer_files, _ = datafold_read(datalist=datalist, basedir=dataroot, fold=-1, key=data_key)
            elif not hasattr(self, "rank") or self.rank == 0:
                logger.info(f"Datalist file has no testing key - {data_key}. No data for inference is specified")

        else:
            raise ValueError("Unsupported parameter type")

    def ensemble_pred(self, preds, sigmoid=False):
        """
        ensemble the results using either "mean" or "vote" method

        Args:
            preds: a list of probability prediction in Tensor-Like format.
            sigmoid: use the sigmoid function to threshold probability one-hot map,
                otherwise argmax is used. Defaults to False

        Returns:
            a tensor which is the ensembled prediction.
        """

        if any(not p.is_cuda for p in preds):
            preds = [p.cpu() for p in preds]  # ensure CPU if at least one is on CPU

        if self.mode == "mean":
            prob = MeanEnsemble()(preds)
            return prob2class(cast(torch.Tensor, prob), dim=0, keepdim=True, sigmoid=sigmoid)
        elif self.mode == "vote":
            classes = [prob2class(p, dim=0, keepdim=True, sigmoid=sigmoid) for p in preds]
            if sigmoid:
                return VoteEnsemble()(classes)  # do not specify num_classes for one-hot encoding
            else:
                return VoteEnsemble(num_classes=preds[0].shape[0])(classes)

    def _apply_algo_specific_param(self, algo_spec_param: dict, param: dict, algo_name: str) -> dict:
        """
        Apply the model-specific params to the prediction params based on the name of the Algo.

        Args:
            algo_spec_param: a dict that has structure of {"<name of algo>": "<pred_params for that algo>"}.
            param: the prediction params to override.
            algo_name: name of the Algo

        Returns:
            param after being updated with the model-specific param
        """
        _param_to_override = deepcopy(algo_spec_param)
        _param = deepcopy(param)
        for k, v in _param_to_override.items():
            if k.lower() == algo_name.lower():
                _param.update(v)
        return _param

    def __call__(self, pred_param: dict | None = None) -> list:
        """
        Use the ensembled model to predict result.

        Args:
            pred_param: prediction parameter dictionary. The key has two groups: the first one will be consumed
                in this function, and the second group will be passed to the `InferClass` to override the
                parameters of the class functions.
                The first group contains:

                    - ``"infer_files"``: file paths to the images to read in a list.
                    - ``"files_slices"``: a value type of `slice`. The files_slices will slice the ``"infer_files"`` and
                      only make prediction on the infer_files[file_slices].
                    - ``"mode"``: ensemble mode. Currently "mean" and "vote" (majority voting) schemes are supported.
                    - ``"image_save_func"``: a dictionary used to instantiate the ``SaveImage`` transform. When specified,
                      the ensemble prediction will save the prediction files, instead of keeping the files in the memory.
                      Example: `{"_target_": "SaveImage", "output_dir": "./"}`
                    - ``"sigmoid"``: use the sigmoid function (e.g. x > 0.5) to convert the prediction probability map
                      to the label class prediction, otherwise argmax(x) is used.
                    - ``"algo_spec_params"``: a dictionary to add pred_params that are specific to a model.
                      The dict has a format of {"<name of algo>": "<pred_params for that algo>"}.

                The parameters in the second group is defined in the ``config`` of each Algo templates. Please check:
                https://github.com/Project-MONAI/research-contributions/tree/main/auto3dseg/algorithm_templates

        Returns:
            A list of tensors or file paths, depending on whether ``"image_save_func"`` is set.
        """
        param = {} if pred_param is None else deepcopy(pred_param)
        files = self.infer_files

        if "infer_files" in param:
            files = param.pop("infer_files")

        if "files_slices" in param:
            slices = param.pop("files_slices")
            files = files[slices]

        if "mode" in param:
            mode = param.pop("mode")
            self.mode = look_up_option(mode, supported=["mean", "vote"])

        sigmoid = param.pop("sigmoid", False)

        if "image_save_func" in param:
            img_saver = ConfigParser(param["image_save_func"]).get_parsed_content()

        algo_spec_params = param.pop("algo_spec_params", {})

        outputs = []
        for _, file in (
            enumerate(tqdm(files, desc="Ensembling (rank 0)..."))
            if has_tqdm and pred_param and pred_param.get("rank", 0) == 0
            else enumerate(files)
        ):
            preds = []
            for algo in self.algo_ensemble:
                infer_algo_name = get_name_from_algo_id(algo[AlgoKeys.ID])
                infer_instance = algo[AlgoKeys.ALGO]
                _param = self._apply_algo_specific_param(algo_spec_params, param, infer_algo_name)
                pred = infer_instance.predict(predict_files=[file], predict_params=_param)
                preds.append(pred[0])
            if "image_save_func" in param:
                try:
                    ensemble_preds = self.ensemble_pred(preds, sigmoid=sigmoid)
                except BaseException:
                    ensemble_preds = self.ensemble_pred([_.to("cpu") for _ in preds], sigmoid=sigmoid)
                res = img_saver(ensemble_preds)
                # res is the path to the saved results
                if hasattr(res, "meta") and "saved_to" in res.meta.keys():
                    res = res.meta["saved_to"]
                else:
                    warn("Image save path not returned.")
                    res = None
            else:
                warn("Prediction returned in list instead of disk, provide image_save_func to avoid out of memory.")
                res = self.ensemble_pred(preds, sigmoid=sigmoid)
            outputs.append(res)
        return outputs

    @abstractmethod
    def collect_algos(self, *args, **kwargs):
        raise NotImplementedError


class AlgoEnsembleBestN(AlgoEnsemble):
    """
    Ensemble method that select N model out of all using the models' best_metric scores

    Args:
        n_best: number of models to pick for ensemble (N).
    """

    def __init__(self, n_best: int = 5):
        super().__init__()
        self.n_best = n_best

    def sort_score(self):
        """
        Sort the best_metrics
        """
        scores = concat_val_to_np(self.algos, [AlgoKeys.SCORE])
        return np.argsort(scores).tolist()

    def collect_algos(self, n_best: int = -1) -> None:
        """
        Rank the algos by finding the top N (n_best) validation scores.
        """

        if n_best <= 0:
            n_best = self.n_best

        ranks = self.sort_score()
        if len(ranks) < n_best:
            warn(f"Found {len(ranks)} available algos (pre-defined n_best={n_best}). All {len(ranks)} will be used.")
            n_best = len(ranks)

        # get the ranks for which the indices are lower than N-n_best
        indices = [r for (i, r) in enumerate(ranks) if i < (len(ranks) - n_best)]

        # remove the found indices
        indices = sorted(indices, reverse=True)

        self.algo_ensemble = deepcopy(self.algos)
        for idx in indices:
            if idx < len(self.algo_ensemble):
                self.algo_ensemble.pop(idx)


class AlgoEnsembleBestByFold(AlgoEnsemble):
    """
    Ensemble method that select the best models that are the tops in each fold.

    Args:
        n_fold: number of cross-validation folds used in training
    """

    def __init__(self, n_fold: int = 5):
        super().__init__()
        self.n_fold = n_fold

    def collect_algos(self) -> None:
        """
        Rank the algos by finding the best model in each cross-validation fold
        """

        self.algo_ensemble = []
        for f_idx in range(self.n_fold):
            best_score = -1.0
            best_model: BundleAlgo | None = None
            for algo in self.algos:
                # algorithm folder: {net}_{fold_index}_{other}
                identifier = algo[AlgoKeys.ID].split("_")[1]
                try:
                    algo_id = int(identifier)
                except ValueError as err:
                    raise ValueError(f"model identifier {identifier} is not number.") from err
                if algo_id == f_idx and algo[AlgoKeys.SCORE] > best_score:
                    best_model = algo
                    best_score = algo[AlgoKeys.SCORE]
            self.algo_ensemble.append(best_model)


class AlgoEnsembleBuilder:
    """
    Build ensemble workflow from configs and arguments.

    Args:
        history: a collection of trained bundleAlgo algorithms.
        data_src_cfg_name: filename of the data source.

    Examples:

        .. code-block:: python

            builder = AlgoEnsembleBuilder(history, data_src_cfg)
            builder.set_ensemble_method(BundleAlgoEnsembleBestN(3))
            ensemble = builder.get_ensemble()

    """

    def __init__(self, history: Sequence[dict[str, Any]], data_src_cfg_name: str | None = None):
        self.infer_algos: list[dict[AlgoKeys, Any]] = []
        self.ensemble: AlgoEnsemble
        self.data_src_cfg = ConfigParser(globals=False)

        if data_src_cfg_name is not None and os.path.exists(str(data_src_cfg_name)):
            self.data_src_cfg.read_config(data_src_cfg_name)

        for algo_dict in history:
            # load inference_config_paths

            name = algo_dict[AlgoKeys.ID]
            gen_algo = algo_dict[AlgoKeys.ALGO]

            best_metric = gen_algo.get_score()
            algo_path = gen_algo.output_path
            infer_path = os.path.join(algo_path, "scripts", "infer.py")

            if not os.path.isdir(algo_path):
                warn(f"{gen_algo.output_path} is not a directory. Please check the path.")

            if not os.path.isfile(infer_path):
                warn(f"{infer_path} is not found. Please check the path.")

            self.add_inferer(name, gen_algo, best_metric)

    def add_inferer(self, identifier: str, gen_algo: BundleAlgo, best_metric: float | None = None) -> None:
        """
        Add model inferer to the builder.

        Args:
            identifier: name of the bundleAlgo.
            gen_algo: a trained BundleAlgo model object.
            best_metric: the best metric in validation of the trained model.
        """

        if best_metric is None:
            raise ValueError("Feature to re-validate is to be implemented")

        algo = {AlgoKeys.ID: identifier, AlgoKeys.ALGO: gen_algo, AlgoKeys.SCORE: best_metric}
        self.infer_algos.append(algo)

    def set_ensemble_method(self, ensemble: AlgoEnsemble, *args: Any, **kwargs: Any) -> None:
        """
        Set the ensemble method.

        Args:
            ensemble: the AlgoEnsemble to build.
        """

        ensemble.set_algos(self.infer_algos)
        ensemble.collect_algos(*args, **kwargs)
        ensemble.set_infer_files(self.data_src_cfg["dataroot"], self.data_src_cfg["datalist"])

        self.ensemble = ensemble

    def get_ensemble(self):
        """Get the ensemble"""

        return self.ensemble


class EnsembleRunner:
    """
    The Runner for ensembler. It ensembles predictions and saves them to the disk with a support of using multi-GPU.

    Args:
        data_src_cfg_name: filename of the data source.
        work_dir: working directory to save the intermediate and final results. Default is `./work_dir`.
        num_fold: number of fold. Default is 5.
        ensemble_method_name: method to ensemble predictions from different model. Default is AlgoEnsembleBestByFold.
                              Supported methods: ["AlgoEnsembleBestN", "AlgoEnsembleBestByFold"].
        mgpu: if using multi-gpu. Default is True.
        kwargs: additional image writing, ensembling parameters and prediction parameters for the ensemble inference.
              - for image saving, please check the supported parameters in SaveImage transform.
              - for prediction parameters, please check the supported parameters in the ``AlgoEnsemble`` callables.
              - for ensemble parameters, please check the documentation of the selected AlgoEnsemble callable.

    Example:

        .. code-block:: python

            ensemble_runner = EnsembleRunner(data_src_cfg_name,
                                             work_dir,
                                             ensemble_method_name,
                                             mgpu=device_setting['n_devices']>1,
                                             **kwargs,
                                             **pred_params)
            ensemble_runner.run(device_setting)

    """

    def __init__(
        self,
        data_src_cfg_name: str,
        work_dir: str = "./work_dir",
        num_fold: int = 5,
        ensemble_method_name: str = "AlgoEnsembleBestByFold",
        mgpu: bool = True,
        **kwargs: Any,
    ) -> None:
        self.data_src_cfg_name = data_src_cfg_name
        self.work_dir = work_dir
        self.num_fold = num_fold
        self.ensemble_method_name = ensemble_method_name
        self.mgpu = mgpu
        self.kwargs = deepcopy(kwargs)
        self.rank = 0
        self.world_size = 1
        self.device_setting: dict[str, int | str] = {
            "CUDA_VISIBLE_DEVICES": ",".join([str(x) for x in range(torch.cuda.device_count())]),
            "n_devices": torch.cuda.device_count(),
            "NUM_NODES": int(os.environ.get("NUM_NODES", 1)),
            "MN_START_METHOD": os.environ.get("MN_START_METHOD", "bcprun"),
            "CMD_PREFIX": os.environ.get("CMD_PREFIX", ""),
        }

    def set_ensemble_method(self, ensemble_method_name: str = "AlgoEnsembleBestByFold", **kwargs: Any) -> None:
        """
        Set the bundle ensemble method

        Args:
            ensemble_method_name: the name of the ensemble method. Only two methods are supported "AlgoEnsembleBestN"
                and "AlgoEnsembleBestByFold".
            kwargs: the keyword arguments used to define the ensemble method. Currently only ``n_best`` for
                ``AlgoEnsembleBestN`` is supported.

        """
        self.ensemble_method_name = look_up_option(
            ensemble_method_name, supported=["AlgoEnsembleBestN", "AlgoEnsembleBestByFold"]
        )
        if self.ensemble_method_name == "AlgoEnsembleBestN":
            n_best = kwargs.pop("n_best", 2)
            self.ensemble_method = AlgoEnsembleBestN(n_best=n_best)
        elif self.ensemble_method_name == "AlgoEnsembleBestByFold":
            self.ensemble_method = AlgoEnsembleBestByFold(n_fold=self.num_fold)  # type: ignore
        else:
            raise NotImplementedError(f"Ensemble method {self.ensemble_method_name} is not implemented.")

    def _pop_kwargs_to_get_image_save_transform(self, **kwargs):
        """
        Pop the kwargs used to define ImageSave class for the ensemble output.

        Args:
            kwargs: image writing parameters for the ensemble inference. The kwargs format follows SaveImage
                transform. For more information, check https://docs.monai.io/en/stable/transforms.html#saveimage .

        Returns:
            save_image: a dictionary that can be used to instantiate a SaveImage class in ConfigParser.
        """

        output_dir = kwargs.pop("output_dir", None)

        if output_dir is None:
            output_dir = os.path.join(self.work_dir, "ensemble_output")
            logger.info(f"The output_dir is not specified. {output_dir} will be used to save ensemble predictions.")

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Directory {output_dir} is created to save ensemble predictions")

        input_yaml = ConfigParser.load_config_file(self.data_src_cfg_name)
        data_root_dir = input_yaml.get("dataroot", "")

        save_image = {
            "_target_": "SaveImage",
            "output_dir": output_dir,
            "output_postfix": kwargs.pop("output_postfix", "ensemble"),
            "output_dtype": kwargs.pop("output_dtype", "$np.uint8"),
            "resample": kwargs.pop("resample", False),
            "print_log": False,
            "savepath_in_metadict": True,
            "data_root_dir": kwargs.pop("data_root_dir", data_root_dir),
            "separate_folder": kwargs.pop("separate_folder", False),
        }

        are_all_args_save_image, extra_args = check_kwargs_exist_in_class_init(SaveImage, kwargs)
        if are_all_args_save_image:
            save_image.update(kwargs)
        else:
            # kwargs has extra values for other purposes, for example, pred_params
            for args in list(kwargs):
                if args not in extra_args:
                    save_image.update({args: kwargs.pop(args)})

        return save_image

    def set_image_save_transform(self, **kwargs: Any) -> None:
        """
        Set the ensemble output transform.

        Args:
            kwargs: image writing parameters for the ensemble inference. The kwargs format follows SaveImage
                transform. For more information, check https://docs.monai.io/en/stable/transforms.html#saveimage .

        """
        are_all_args_present, extra_args = check_kwargs_exist_in_class_init(SaveImage, kwargs)
        if are_all_args_present:
            self.kwargs.update(kwargs)
        else:
            raise ValueError(
                f"{extra_args} are not supported in monai.transforms.SaveImage,"
                "Check https://docs.monai.io/en/stable/transforms.html#saveimage for more information."
            )

    def set_num_fold(self, num_fold: int = 5) -> None:
        """
        Set the number of cross validation folds for all algos.

        Args:
            num_fold: a positive integer to define the number of folds.
        """

        if num_fold <= 0:
            raise ValueError(f"num_fold is expected to be an integer greater than zero. Now it gets {num_fold}")
        self.num_fold = num_fold

    def ensemble(self):
        if self.mgpu:  # torch.cuda.device_count() is not used because env is not set by autorunner
            # init multiprocessing and update infer_files
            dist.init_process_group(backend="nccl", init_method="env://")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.addFilter(RankFilter())
        # set params after init_process_group to know the rank
        self.set_num_fold(num_fold=self.num_fold)
        self.set_ensemble_method(self.ensemble_method_name, **self.kwargs)
        # self.kwargs needs to pop out args for set_image_save_transform
        save_image = self._pop_kwargs_to_get_image_save_transform(**self.kwargs)

        history = import_bundle_algo_history(self.work_dir, only_trained=False)
        history_untrained = [h for h in history if not h[AlgoKeys.IS_TRAINED]]
        if history_untrained:
            logger.warning(
                f"Ensembling step will skip {[h[AlgoKeys.ID] for h in history_untrained]} untrained algos."
                "Generally it means these algos did not complete training."
            )
            history = [h for h in history if h[AlgoKeys.IS_TRAINED]]
        if len(history) == 0:
            raise ValueError(
                f"Could not find the trained results in {self.work_dir}. "
                "Possibly the required training step was not completed."
            )

        builder = AlgoEnsembleBuilder(history, self.data_src_cfg_name)
        builder.set_ensemble_method(self.ensemble_method)
        self.ensembler = builder.get_ensemble()
        infer_files = self.ensembler.infer_files
        if len(infer_files) < self.world_size:
            if len(infer_files) == 0:
                logger.info("No testing files for inference is provided. Ensembler ending.")
                return
            infer_files = [infer_files[self.rank]] if self.rank < len(infer_files) else []
        else:
            infer_files = partition_dataset(
                data=infer_files, shuffle=False, num_partitions=self.world_size, even_divisible=False
            )[self.rank]

        # TO DO: Add some function in ensembler for infer_files update?
        self.ensembler.infer_files = infer_files
        # add rank to pred_params
        self.kwargs["rank"] = self.rank
        self.kwargs["image_save_func"] = save_image
        logger.info("Auto3Dseg picked the following networks to ensemble:")
        for algo in self.ensembler.get_algo_ensemble():
            logger.info(algo[AlgoKeys.ID])
        output_dir = save_image["output_dir"]
        logger.info(f"Auto3Dseg ensemble prediction outputs will be saved in {output_dir}.")
        self.ensembler(pred_param=self.kwargs)

        if self.mgpu:
            dist.destroy_process_group()

    def run(self, device_setting: dict | None = None) -> None:
        """
        Load the run function in the training script of each model. Training parameter is predefined by the
        algo_config.yaml file, which is pre-filled by the fill_template_config function in the same instance.

        Args:
            device_setting: device related settings, should follow the device_setting in auto_runner.set_device_info.
                'CUDA_VISIBLE_DEVICES' should be a string e.g. '0,1,2,3'
        """
        # device_setting set default value and sanity check, in case device_setting not from autorunner
        if device_setting is not None:
            self.device_setting.update(device_setting)
            self.device_setting["n_devices"] = len(str(self.device_setting["CUDA_VISIBLE_DEVICES"]).split(","))
        self._create_cmd()

    def _create_cmd(self) -> None:
        if int(self.device_setting["NUM_NODES"]) <= 1 and int(self.device_setting["n_devices"]) <= 1:
            # if single GPU
            logger.info("Ensembling using single GPU!")
            self.ensemble()
            return

        # define base cmd for subprocess
        base_cmd = f"monai.apps.auto3dseg EnsembleRunner ensemble \
                --data_src_cfg_name {self.data_src_cfg_name} \
                --work_dir {self.work_dir} \
                --num_fold {self.num_fold} \
                --ensemble_method_name {self.ensemble_method_name} \
                --mgpu True"

        if self.kwargs and isinstance(self.kwargs, Mapping):
            for k, v in self.kwargs.items():
                base_cmd += f" --{k}={v}"
        # define env for subprocess
        ps_environ = os.environ.copy()
        ps_environ["CUDA_VISIBLE_DEVICES"] = str(self.device_setting["CUDA_VISIBLE_DEVICES"])
        if int(self.device_setting["NUM_NODES"]) > 1:
            if self.device_setting["MN_START_METHOD"] != "bcprun":
                raise NotImplementedError(
                    f"{self.device_setting['MN_START_METHOD']} is not supported yet. "
                    "Try modify EnsembleRunner._create_cmd for your cluster."
                )
            logger.info(f"Ensembling on {self.device_setting['NUM_NODES']} nodes!")
            cmd = _prepare_cmd_bcprun("-m " + base_cmd, cmd_prefix=f"{self.device_setting['CMD_PREFIX']}")
            _run_cmd_bcprun(cmd, n=self.device_setting["NUM_NODES"], p=self.device_setting["n_devices"])

        else:
            logger.info(f"Ensembling using {self.device_setting['n_devices']} GPU!")
            cmd = _prepare_cmd_torchrun("-m " + base_cmd)
            _run_cmd_torchrun(
                cmd, nnodes=1, nproc_per_node=self.device_setting["n_devices"], env=ps_environ, check=True
            )
        return

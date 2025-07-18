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
import shutil
import warnings
from copy import deepcopy
from time import sleep
from typing import Any, cast

import torch

from monai.apps.auto3dseg.bundle_gen import BundleGen
from monai.apps.auto3dseg.data_analyzer import DataAnalyzer
from monai.apps.auto3dseg.ensemble_builder import EnsembleRunner
from monai.apps.auto3dseg.hpo_gen import NNIGen
from monai.apps.auto3dseg.utils import export_bundle_algo_history, import_bundle_algo_history
from monai.apps.utils import get_logger
from monai.auto3dseg.utils import algo_to_pickle
from monai.bundle import ConfigParser
from monai.transforms import SaveImage
from monai.utils import AlgoKeys, has_option, look_up_option, optional_import
from monai.utils.misc import check_kwargs_exist_in_class_init, run_cmd

logger = get_logger(module_name=__name__)

nni, has_nni = optional_import("nni")


class AutoRunner:
    """
    An interface for handling Auto3Dseg with minimal inputs and understanding of the internal states in Auto3Dseg.
    The users can run the Auto3Dseg with default settings in one line of code. They can also customize the advanced
    features Auto3Dseg in a few additional lines. Examples of customization include

        - change cross-validation folds
        - change training/prediction parameters
        - change ensemble methods
        - automatic hyperparameter optimization.

    The output of the interface is a directory that contains

        - data statistics analysis report
        - algorithm definition files (scripts, configs, pickle objects) and training results (checkpoints, accuracies)
        - the predictions on the testing datasets from the final algorithm ensemble
        - a copy of the input arguments in form of YAML
        - cached intermediate results

    Args:
        work_dir: working directory to save the intermediate and final results.
        input: the configuration dictionary or the file path to the configuration in form of YAML.
            The configuration should contain datalist, dataroot, modality, multigpu, and class_names info.
        algos: optionally specify algorithms to use.  If a dictionary, must be in the form
            {"algname": dict(_target_="algname.scripts.algo.AlgnameAlgo", template_path="algname"), ...}
            If a list or a string, defines a subset of names of the algorithms to use, e.g. 'segresnet' or
            ['segresnet', 'dints'] out of the full set of algorithm templates provided by templates_path_or_url.
            Defaults to None, to use all available algorithms.
        analyze: on/off switch to run DataAnalyzer and generate a datastats report. Defaults to None, to automatically
            decide based on cache, and run data analysis only if we have not completed this step yet.
        algo_gen: on/off switch to run AlgoGen and generate templated BundleAlgos. Defaults to None, to automatically
            decide based on cache, and run algorithm folders generation only if we have not completed this step yet.
        train: on/off switch to run training and generate algorithm checkpoints. Defaults to None, to automatically
            decide based on cache, and run training only if we have not completed this step yet.
        hpo: use hyperparameter optimization (HPO) in the training phase. Users can provide a list of
            hyper-parameter and a search will be performed to investigate the algorithm performances.
        hpo_backend: a string that indicates the backend of the HPO. Currently, only NNI Grid-search mode
            is supported
        ensemble: on/off switch to run model ensemble and use the ensemble to predict outputs in testing
            datasets.
        not_use_cache: if the value is True, it will ignore all cached results in data analysis,
            algorithm generation, or training, and start the pipeline from scratch.
        templates_path_or_url: the folder with the algorithm templates or a url. If None provided, the default template
            zip url will be downloaded and extracted into the work_dir.
        allow_skip: a switch passed to BundleGen process which determines if some Algo in the default templates
            can be skipped based on the analysis on the dataset from Auto3DSeg DataAnalyzer.
        mlflow_tracking_uri: a tracking URI for MLflow server which could be local directory or address of the remote
            tracking Server; MLflow runs will be recorded locally in algorithms' model folder if the value is None.
        mlflow_experiment_name: the name of the experiment in MLflow server.
        kwargs: image writing parameters for the ensemble inference. The kwargs format follows the SaveImage
            transform. For more information, check https://docs.monai.io/en/stable/transforms.html#saveimage.


    Examples:
        - User can use the one-liner to start the Auto3Dseg workflow

        .. code-block:: bash

            python -m monai.apps.auto3dseg AutoRunner run --input \
            '{"modality": "ct", "datalist": "dl.json", "dataroot": "/dr", "multigpu": true, "class_names": ["A", "B"]}'

        - User can also save the input dictionary as a input YAML file and use the following one-liner

        .. code-block:: bash

            python -m monai.apps.auto3dseg AutoRunner run --input=./input.yaml

        - User can specify work_dir and data source config input and run AutoRunner:

        .. code-block:: python

            work_dir = "./work_dir"
            input = "path/to/input_yaml"
            runner = AutoRunner(work_dir=work_dir, input=input)
            runner.run()

        - User can specify a subset of algorithms to use and run AutoRunner:

        .. code-block:: python

            work_dir = "./work_dir"
            input = "path/to/input_yaml"
            algos = ["segresnet", "dints"]
            runner = AutoRunner(work_dir=work_dir, input=input, algos=algos)
            runner.run()

        - User can specify a local folder with algorithms templates and run AutoRunner:

        .. code-block:: python

            work_dir = "./work_dir"
            input = "path/to/input_yaml"
            algos = "segresnet"
            templates_path_or_url = "./local_path_to/algorithm_templates"
            runner = AutoRunner(work_dir=work_dir, input=input, algos=algos, templates_path_or_url=templates_path_or_url)
            runner.run()

        - User can specify training parameters by:

        .. code-block:: python

            input = "path/to/input_yaml"
            runner = AutoRunner(input=input)
            train_param = {
                "num_epochs_per_validation": 1,
                "num_images_per_batch": 2,
                "num_epochs": 2,
            }
            runner.set_training_params(params=train_param)  # 2 epochs
            runner.run()

        - User can specify the fold number of cross validation

        .. code-block:: python

            input = "path/to/input_yaml"
            runner = AutoRunner(input=input)
            runner.set_num_fold(n_fold = 2)
            runner.run()

        - User can specify the prediction parameters during algo ensemble inference:

        .. code-block:: python

            input = "path/to/input_yaml"
            pred_params = {
                'files_slices': slice(0,2),
                'mode': "vote",
                'sigmoid': True,
            }
            runner = AutoRunner(input=input)
            runner.set_prediction_params(params=pred_params)
            runner.run()

        - User can define a grid search space and use the HPO during training.

        .. code-block:: python

            input = "path/to/input_yaml"
            runner = AutoRunner(input=input, hpo=True)
            runner.set_nni_search_space({"learning_rate": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]}})
            runner.run()

    Notes:
        Expected results in the work_dir as below::

            work_dir/
            ├── algorithm_templates # bundle algo templates (scripts/configs)
            ├── cache.yaml          # Autorunner will automatically cache results to save time
            ├── datastats.yaml      # datastats of the dataset
            ├── dints_0             # network scripts/configs/checkpoints and pickle object of the algo
            ├── ensemble_output     # the prediction of testing datasets from the ensemble of the algos
            ├── input.yaml          # copy of the input data source configs
            ├── segresnet_0         # network scripts/configs/checkpoints and pickle object of the algo
            ├── segresnet2d_0       # network scripts/configs/checkpoints and pickle object of the algo
            └── swinunetr_0         # network scripts/configs/checkpoints and pickle object of the algo

    """

    analyze_params: dict | None

    def __init__(
        self,
        work_dir: str = "./work_dir",
        input: dict[str, Any] | str | None = None,
        algos: dict | list | str | None = None,
        analyze: bool | None = None,
        algo_gen: bool | None = None,
        train: bool | None = None,
        hpo: bool = False,
        hpo_backend: str = "nni",
        ensemble: bool = True,
        not_use_cache: bool = False,
        templates_path_or_url: str | None = None,
        allow_skip: bool = True,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str | None = None,
        **kwargs: Any,
    ):
        if input is None and os.path.isfile(os.path.join(os.path.abspath(work_dir), "input.yaml")):
            input = os.path.join(os.path.abspath(work_dir), "input.yaml")
            logger.info(f"Input config is not provided, using the default {input}")

        self.data_src_cfg = dict()
        if isinstance(input, dict):
            self.data_src_cfg = input
        elif isinstance(input, str) and os.path.isfile(input):
            self.data_src_cfg = ConfigParser.load_config_file(input)
            logger.info(f"Loading input config {input}")
        else:
            raise ValueError(f"{input} is not a valid file or dict")

        if "work_dir" in self.data_src_cfg:  # override from config
            work_dir = self.data_src_cfg["work_dir"]
        self.work_dir = os.path.abspath(work_dir)

        logger.info(f"AutoRunner using work directory {self.work_dir}")
        os.makedirs(self.work_dir, exist_ok=True)
        self.data_src_cfg_name = os.path.join(self.work_dir, "input.yaml")

        self.algos = algos
        self.templates_path_or_url = templates_path_or_url
        self.allow_skip = allow_skip

        # cache.yaml
        self.not_use_cache = not_use_cache
        self.cache_filename = os.path.join(self.work_dir, "cache.yaml")
        self.cache = self.read_cache()
        self.export_cache()

        # determine if we need to analyze, algo_gen or train from cache, unless manually provided
        self.analyze = not self.cache["analyze"] if analyze is None else analyze
        self.algo_gen = not self.cache["algo_gen"] if algo_gen is None else algo_gen
        self.train = train
        self.ensemble = ensemble  # last step, no need to check
        self.hpo = hpo and has_nni
        self.hpo_backend = hpo_backend
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.kwargs = deepcopy(kwargs)

        # parse input config for AutoRunner param overrides
        for param in [
            "analyze",
            "algo_gen",
            "train",
            "hpo",
            "ensemble",
            "not_use_cache",
            "allow_skip",
        ]:  # override from config
            if param in self.data_src_cfg and isinstance(self.data_src_cfg[param], bool):
                setattr(self, param, self.data_src_cfg[param])  # e.g. self.analyze = self.data_src_cfg["analyze"]

        for param in [
            "algos",
            "hpo_backend",
            "templates_path_or_url",
            "mlflow_tracking_uri",
            "mlflow_experiment_name",
        ]:  # override from config
            if param in self.data_src_cfg:
                setattr(self, param, self.data_src_cfg[param])  # e.g. self.algos = self.data_src_cfg["algos"]

        missing_keys = {"dataroot", "datalist", "modality"}.difference(self.data_src_cfg.keys())
        if len(missing_keys) > 0:
            raise ValueError(f"Config keys are missing {missing_keys}")

        if not os.path.exists(self.data_src_cfg["datalist"]):
            raise ValueError(f"Datalist file is not found {self.data_src_cfg['datalist']}")

        # copy datalist to work_dir
        datalist_filename = os.path.join(self.work_dir, os.path.basename(self.data_src_cfg["datalist"]))
        if datalist_filename != self.data_src_cfg["datalist"]:
            try:
                shutil.copyfile(self.data_src_cfg["datalist"], datalist_filename)
                logger.info(f"Datalist was copied to work_dir: {datalist_filename}")
            except shutil.SameFileError:
                pass

        # inspect and update folds
        self.max_fold = self.inspect_datalist_folds(datalist_filename=datalist_filename)
        if "num_fold" in self.data_src_cfg:
            num_fold = int(self.data_src_cfg["num_fold"])  # override from config
            logger.info(f"Setting num_fold {num_fold} based on the input config.")
        else:
            num_fold = self.max_fold
            logger.info(f"Setting num_fold {num_fold} based on the input datalist {datalist_filename}.")

        self.data_src_cfg["datalist"] = datalist_filename  # update path to a version in work_dir and save user input
        ConfigParser.export_config_file(
            config=self.data_src_cfg, filepath=self.data_src_cfg_name, fmt="yaml", sort_keys=False
        )

        self.dataroot = self.data_src_cfg["dataroot"]
        self.datastats_filename = os.path.join(self.work_dir, "datastats.yaml")
        self.datalist_filename = datalist_filename

        self.set_training_params()
        self.set_device_info()
        self.set_prediction_params()
        self.set_analyze_params()
        self.set_ensemble_method()
        self.set_num_fold(num_fold=num_fold)

        self.gpu_customization = False
        self.gpu_customization_specs: dict[str, Any] = {}

        # hpo
        if self.hpo_backend.lower() != "nni":
            raise NotImplementedError("HPOGen backend only supports NNI")
        self.hpo = self.hpo and has_nni
        self.set_hpo_params()
        self.search_space: dict[str, dict[str, Any]] = {}
        self.hpo_tasks = 0

        if "sigmoid" not in self.kwargs and "sigmoid" in self.data_src_cfg:
            self.kwargs["sigmoid"] = self.data_src_cfg["sigmoid"]

    def read_cache(self):
        """
        Check if the intermediate result is cached after each step in the current working directory

        Returns:
            a dict of cache results. If not_use_cache is set to True, or there is no cache file in the
            working directory, the result will be ``empty_cache`` in which all ``has_cache`` keys are
            set to False.
        """

        empty_cache = {"analyze": False, "datastats": None, "algo_gen": False, "train": False}

        if self.not_use_cache or not os.path.isfile(self.cache_filename):
            return empty_cache

        cache = ConfigParser.load_config_file(self.cache_filename)

        for k, v in empty_cache.items():
            cache.setdefault(k, v)

        if cache["analyze"]:
            if not (isinstance(cache["datastats"], str) and os.path.isfile(cache["datastats"])):
                cache["analyze"] = False
                cache["datastats"] = None

        if cache["algo_gen"]:
            history = import_bundle_algo_history(self.work_dir, only_trained=False)
            if len(history) == 0:  # no saved algo_objects
                cache["algo_gen"] = False

        if cache["train"]:
            trained_history = import_bundle_algo_history(self.work_dir, only_trained=True)
            if len(trained_history) == 0:
                cache["train"] = False

        return cache

    def export_cache(self, **kwargs):
        """
        Save the cache state as ``cache.yaml`` in the working directory
        """
        self.cache.update(kwargs)
        ConfigParser.export_config_file(
            self.cache, self.cache_filename, fmt="yaml", default_flow_style=None, sort_keys=False
        )

    def inspect_datalist_folds(self, datalist_filename: str) -> int:
        """
        Returns number of folds in the datalist file, and assigns fold numbers if not provided.

        Args:
            datalist_filename: path to the datalist file.

        Notes:
            If the fold key is not provided, it auto generates 5 folds assignments in the training key list.
            If validation key list is available, then it assumes a single fold validation.
        """

        datalist = ConfigParser.load_config_file(datalist_filename)
        if "training" not in datalist:
            raise ValueError("Datalist files has no training key:" + str(datalist_filename))

        fold_list = [int(d["fold"]) for d in datalist["training"] if "fold" in d]

        if len(fold_list) > 0:
            num_fold = max(fold_list) + 1
            logger.info(f"Found num_fold {num_fold} based on the input datalist {datalist_filename}.")
            # check if every fold is present
            if len(set(fold_list)) != num_fold:
                raise ValueError(f"Fold numbers are not continuous from 0 to {num_fold - 1}")
        elif "validation" in datalist and len(datalist["validation"]) > 0:
            logger.info("No fold numbers provided, attempting to use a single fold based on the validation key")
            # update the datalist file
            for d in datalist["training"]:
                d["fold"] = 1
            for d in datalist["validation"]:
                d["fold"] = 0

            val_labels = {d["label"]: d for d in datalist["validation"] if "label" in d}
            logger.info(
                f"Found {len(val_labels)} items in the validation key, saving updated datalist to", datalist_filename
            )

            # check for duplicates
            for d in datalist["training"]:
                if d["label"] in val_labels:
                    d["fold"] = 0
                    del val_labels[d["label"]]

            datalist["training"] = datalist["training"] + list(val_labels.values())

            ConfigParser.export_config_file(datalist, datalist_filename, fmt="json", indent=4)
            num_fold = 1

        else:
            num_fold = 5

            warnings.warn(
                f"Datalist has no folds specified {datalist_filename}..."
                f"Generating {num_fold} folds randomly."
                f"Please consider presaving fold numbers beforehand for repeated experiments."
            )

            from sklearn.model_selection import KFold

            kf = KFold(n_splits=num_fold, shuffle=True, random_state=0)
            for i, (_, valid_idx) in enumerate(kf.split(datalist["training"])):
                for vi in valid_idx:
                    datalist["training"][vi]["fold"] = i

            ConfigParser.export_config_file(datalist, datalist_filename, fmt="json", indent=4)

        return num_fold

    def set_gpu_customization(
        self, gpu_customization: bool = False, gpu_customization_specs: dict[str, Any] | None = None
    ) -> AutoRunner:
        """
        Set options for GPU-based parameter customization/optimization.

        Args:
            gpu_customization: the switch to determine automatically customize/optimize bundle script/config
                parameters for each bundleAlgo based on gpus. Custom parameters are obtained through dummy
                training to simulate the actual model training process and hyperparameter optimization (HPO)
                experiments.
            gpu_customization_specs (optional): the dictionary to enable users overwrite the HPO settings. user can
                overwrite part of variables as follows or all of them. The structure is as follows.

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
        self.gpu_customization = gpu_customization
        if gpu_customization_specs is not None:
            self.gpu_customization_specs = gpu_customization_specs

        return self

    def set_num_fold(self, num_fold: int = 5) -> AutoRunner:
        """
        Set the number of cross validation folds for all algos.

        Args:
            num_fold: a positive integer to define the number of folds.
        """

        if num_fold <= 0:
            raise ValueError(f"num_fold is expected to be an integer greater than zero. Now it gets {num_fold}")
        if num_fold > self.max_fold:
            # Auto3DSeg must contain validation set, so the maximum fold number is max_fold.
            raise ValueError(
                f"num_fold is greater than the maximum fold number {self.max_fold} in {self.datalist_filename}."
            )
        self.num_fold = num_fold

        return self

    def set_training_params(self, params: dict[str, Any] | None = None) -> AutoRunner:
        """
        Set the training params for all algos.

        Args:
            params: a dict that defines the overriding key-value pairs during training. The overriding method
                is defined by the algo class.

        Examples:
            For BundleAlgo objects, the training parameter to shorten the training time to a few epochs can be
                {"num_epochs": 2, "num_epochs_per_validation": 1}

        """
        self.train_params = deepcopy(params) if params is not None else {}
        if "CUDA_VISIBLE_DEVICES" in self.train_params:
            warnings.warn(
                "CUDA_VISIBLE_DEVICES is deprecated from 'set_training_params'. Use 'set_device_info' instead.",
                DeprecationWarning,
            )

        return self

    def set_device_info(
        self,
        cuda_visible_devices: list[int] | str | None = None,
        num_nodes: int | None = None,
        mn_start_method: str | None = None,
        cmd_prefix: str | None = None,
    ) -> AutoRunner:
        """
        Set the device related info

        Args:
            cuda_visible_devices: define GPU ids for data analyzer, training, and ensembling.
                List of GPU ids [0,1,2,3] or a string "0,1,2,3".
                Default using env "CUDA_VISIBLE_DEVICES" or all devices available.
            num_nodes: number of nodes for training and ensembling.
                Default using env "NUM_NODES" or 1 if "NUM_NODES" is unset.
            mn_start_method: multi-node start method. Autorunner will use the method to start multi-node processes.
                Default using env "MN_START_METHOD" or 'bcprun' if "MN_START_METHOD" is unset.
            cmd_prefix: command line prefix for subprocess running in BundleAlgo and EnsembleRunner.
                Default using env "CMD_PREFIX" or None, examples are:

                    - single GPU/CPU or multinode bcprun: "python " or "/opt/conda/bin/python3.9 ",
                    - single node multi-GPU running "torchrun --nnodes=1 --nproc_per_node=2 "

                If user define this prefix, please make sure --nproc_per_node matches cuda_visible_device or
                os.env['CUDA_VISIBLE_DEVICES']. Also always set --nnodes=1. Set num_nodes for multi-node.
        """
        self.device_setting: dict[str, Any] = {}
        if cuda_visible_devices is None:
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None:  # still None after reading the environ
            self.device_setting["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in range(torch.cuda.device_count())])
            self.device_setting["n_devices"] = torch.cuda.device_count()
        elif isinstance(cuda_visible_devices, str):
            self.device_setting["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            self.device_setting["n_devices"] = len(cuda_visible_devices.split(","))
        elif isinstance(cuda_visible_devices, (list, tuple)):
            self.device_setting["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cuda_visible_devices])
            self.device_setting["n_devices"] = len(cuda_visible_devices)
        else:
            logger.warning(f"Wrong format of cuda_visible_devices {cuda_visible_devices}, devices not set")

        if num_nodes is None:
            num_nodes = int(os.environ.get("NUM_NODES", 1))
        self.device_setting["NUM_NODES"] = num_nodes

        if mn_start_method is None:
            mn_start_method = os.environ.get("MN_START_METHOD", "bcprun")
        self.device_setting["MN_START_METHOD"] = mn_start_method

        if cmd_prefix is None:
            cmd_prefix = os.environ.get("CMD_PREFIX", "")
        self.device_setting["CMD_PREFIX"] = cmd_prefix

        if cmd_prefix is not None:
            logger.info(f"Using user defined command running prefix {cmd_prefix}, will override other settings")

        return self

    def set_ensemble_method(self, ensemble_method_name: str = "AlgoEnsembleBestByFold", **kwargs: Any) -> AutoRunner:
        """
        Set the bundle ensemble method name and parameters for save image transform parameters.

        Args:
            ensemble_method_name: the name of the ensemble method. Only two methods are supported "AlgoEnsembleBestN"
                and "AlgoEnsembleBestByFold".
            kwargs: the keyword arguments used to define the ensemble method. Currently only ``n_best`` for
                ``AlgoEnsembleBestN`` is supported.
        """
        self.ensemble_method_name = look_up_option(
            ensemble_method_name, supported=["AlgoEnsembleBestN", "AlgoEnsembleBestByFold"]
        )
        self.kwargs.update(kwargs)

        return self

    def set_image_save_transform(self, **kwargs: Any) -> AutoRunner:
        """
        Set the ensemble output transform.

        Args:
            kwargs: image writing parameters for the ensemble inference. The kwargs format follows SaveImage
                transform. For more information, check https://docs.monai.io/en/stable/transforms.html#saveimage.

        """

        are_all_args_present, extra_args = check_kwargs_exist_in_class_init(SaveImage, kwargs)
        if are_all_args_present:
            self.kwargs.update(kwargs)
        else:
            raise ValueError(
                f"{extra_args} are not supported in monai.transforms.SaveImage,"
                "Check https://docs.monai.io/en/stable/transforms.html#saveimage for more information."
            )

        return self

    def set_prediction_params(self, params: dict[str, Any] | None = None) -> AutoRunner:
        """
        Set the prediction params for all algos.

        Args:
            params: a dict that defines the overriding key-value pairs during prediction. The overriding method
                is defined by the algo class.

        Examples:

            For BundleAlgo objects, this set of param will specify the algo ensemble to only inference the first
                two files in the testing datalist {"file_slices": slice(0, 2)}

        """
        self.pred_params = deepcopy(params) if params is not None else {}

        return self

    def set_analyze_params(self, params: dict[str, Any] | None = None) -> AutoRunner:
        """
        Set the data analysis extra params.

        Args:
            params: a dict that defines the overriding key-value pairs during training. The overriding method
                is defined by the algo class.

        """
        if params is None:
            self.analyze_params = {"do_ccp": False, "device": "cuda"}
        else:
            self.analyze_params = deepcopy(params)

        return self

    def set_hpo_params(self, params: dict[str, Any] | None = None) -> AutoRunner:
        """
        Set parameters for the HPO module and the algos before the training. It will attempt to (1) override bundle
        templates with the key-value pairs in ``params`` (2) change the config of the HPO module (e.g. NNI) if the
        key is found to be one of:

            - "trialCodeDirectory"
            - "trialGpuNumber"
            - "trialConcurrency"
            - "maxTrialNumber"
            - "maxExperimentDuration"
            - "tuner"
            - "trainingService"

        and (3) enable the dry-run mode if the user would generate the NNI configs without starting the NNI service.

        Args:
            params: a dict that defines the overriding key-value pairs during instantiation of the algo. For
                BundleAlgo, it will override the template config filling.

        Notes:
            Users can set ``nni_dry_run`` to ``True`` in the ``params`` to enable the dry-run mode for the NNI backend.

        """
        self.hpo_params = self.train_params if params is None else params

        return self

    def set_nni_search_space(self, search_space: dict[str, Any]) -> AutoRunner:
        """
        Set the search space for NNI parameter search.

        Args:
            search_space: hyper parameter search space in the form of dict. For more information, please check
                NNI documentation: https://nni.readthedocs.io/en/v2.2/Tutorial/SearchSpaceSpec.html .
        """
        value_combinations = 1
        for k, v in search_space.items():
            if "_value" not in v:
                raise ValueError(f"{search_space} key {k} value {v} has not _value")
            value_combinations *= len(v["_value"])

        self.search_space = search_space
        self.hpo_tasks = value_combinations

        return self

    def _train_algo_in_sequence(self, history: list[dict[str, Any]]) -> None:
        """
        Train the Algos in a sequential scheme. The order of training is randomized.

        Args:
            history: the history of generated Algos. It is a list of dicts. Each element has the task name
                (e.g. "dints_0" for dints network in fold 0) as the key and the algo object as the value.
                After the training, the algo object with the ``best_metric`` will be saved as a pickle file.

        Note:
            The final results of the model training will be written to all the generated algorithm's output
            folders under the working directory. The results include the model checkpoints, a
            progress.yaml, accuracies in CSV and a pickle file of the Algo object.
        """
        for algo_dict in history:
            algo = algo_dict[AlgoKeys.ALGO]
            if has_option(algo.train, "device_setting"):
                algo.train(self.train_params, self.device_setting)
            else:
                algo.train(self.train_params)
            acc = algo.get_score()

            algo_meta_data = {str(AlgoKeys.SCORE): acc}
            algo_to_pickle(algo, template_path=algo.template_path, **algo_meta_data)

    def _train_algo_in_nni(self, history: list[dict[str, Any]]) -> None:
        """
        Train the Algos using HPO.

        Args:
            history: the history of generated Algos. It is a list of dicts. Each element has the task name
                (e.g. "dints_0" for dints network in fold 0) as the key and the algo object as the value.
                After the training, the algo object with the ``best_metric`` will be saved as a pickle file.

        Note:
            The final results of the model training will not be written to all the previously generated
            algorithm's output folders. Instead, HPO will generate a new algo during the searching, and
            the new algo will be saved under the working directory with a different format of the name.
            For example, if the searching space has "learning_rate", the result of HPO will be written to
            a folder name with original task name and the param (e.g. "dints_0_learning_rate_0.001").
            The results include the model checkpoints, a progress.yaml, accuracies in CSV and a pickle
            file of the Algo object.

        """
        default_nni_config = {
            "trialCodeDirectory": ".",
            "trialGpuNumber": torch.cuda.device_count(),
            "trialConcurrency": 1,
            "maxTrialNumber": 10,
            "maxExperimentDuration": "1h",
            "tuner": {"name": "GridSearch"},
            "trainingService": {"platform": "local", "useActiveGpu": True},
        }

        last_total_tasks = len(import_bundle_algo_history(self.work_dir, only_trained=True))
        mode_dry_run = self.hpo_params.pop("nni_dry_run", False)
        for algo_dict in history:
            name = algo_dict[AlgoKeys.ID]
            algo = algo_dict[AlgoKeys.ALGO]
            nni_gen = NNIGen(algo=algo, params=self.hpo_params)
            obj_filename = nni_gen.get_obj_filename()
            nni_config = deepcopy(default_nni_config)
            # override the default nni config with the same key in hpo_params
            for key in self.hpo_params:
                if key in nni_config:
                    nni_config[key] = self.hpo_params[key]
            nni_config.update({"experimentName": name})
            nni_config.update({"search_space": self.search_space})
            trial_cmd = "python -m monai.apps.auto3dseg NNIGen run_algo " + obj_filename + " " + self.work_dir
            nni_config.update({"trialCommand": trial_cmd})
            nni_config_filename = os.path.abspath(os.path.join(self.work_dir, f"{name}_nni_config.yaml"))
            ConfigParser.export_config_file(nni_config, nni_config_filename, fmt="yaml", default_flow_style=None)

            max_trial = min(self.hpo_tasks, cast(int, default_nni_config["maxTrialNumber"]))
            cmd = "nnictl create --config " + nni_config_filename + " --port 8088"

            if mode_dry_run:
                logger.info(f"AutoRunner HPO is in dry-run mode. Please manually launch: {cmd}")
                continue

            run_cmd(cmd.split(), check=True)

            n_trainings = len(import_bundle_algo_history(self.work_dir, only_trained=True))
            while n_trainings - last_total_tasks < max_trial:
                sleep(1)
                n_trainings = len(import_bundle_algo_history(self.work_dir, only_trained=True))

            cmd = "nnictl stop --all"
            run_cmd(cmd.split(), check=True)
            logger.info(f"NNI completes HPO on {name}")
            last_total_tasks = n_trainings

    def run(self):
        """
        Run the AutoRunner pipeline
        """
        # step 1: data analysis
        if self.analyze and self.analyze_params is not None:
            logger.info("Running data analysis...")
            da = DataAnalyzer(
                self.datalist_filename, self.dataroot, output_path=self.datastats_filename, **self.analyze_params
            )
            da.get_all_case_stats()

            da = None  # type: ignore
            torch.cuda.empty_cache()

            self.export_cache(analyze=True, datastats=self.datastats_filename)
        else:
            logger.info("Skipping data analysis...")

        # step 2: algorithm generation
        if self.algo_gen:
            if not os.path.isfile(self.datastats_filename):
                raise ValueError(
                    f"Could not find the datastats file {self.datastats_filename}. "
                    "Possibly the required data analysis step was not completed."
                )

            bundle_generator = BundleGen(
                algos=self.algos,
                algo_path=self.work_dir,
                templates_path_or_url=self.templates_path_or_url,
                data_stats_filename=self.datastats_filename,
                data_src_cfg_name=self.data_src_cfg_name,
                mlflow_tracking_uri=self.mlflow_tracking_uri,
                mlflow_experiment_name=self.mlflow_experiment_name,
            )

            if self.gpu_customization:
                bundle_generator.generate(
                    self.work_dir,
                    num_fold=self.num_fold,
                    gpu_customization=self.gpu_customization,
                    gpu_customization_specs=self.gpu_customization_specs,
                    allow_skip=self.allow_skip,
                )
            else:
                bundle_generator.generate(self.work_dir, num_fold=self.num_fold, allow_skip=self.allow_skip)
            history = bundle_generator.get_history()
            export_bundle_algo_history(history)
            self.export_cache(algo_gen=True)
        else:
            logger.info("Skipping algorithm generation...")

        # step 3: algo training
        auto_train_choice = self.train is None
        if self.train or (auto_train_choice and not self.cache["train"]):
            history = import_bundle_algo_history(self.work_dir, only_trained=False)

            if len(history) == 0:
                raise ValueError(
                    f"Could not find training scripts in {self.work_dir}. "
                    "Possibly the required algorithms generation step was not completed."
                )

            if auto_train_choice:
                skip_algos = [h[AlgoKeys.ID] for h in history if h[AlgoKeys.IS_TRAINED]]
                if skip_algos:
                    logger.info(
                        f"Skipping already trained algos {skip_algos}."
                        "Set option train=True to always retrain all algos."
                    )
                    history = [h for h in history if not h[AlgoKeys.IS_TRAINED]]

            if len(history) > 0:
                if not self.hpo:
                    self._train_algo_in_sequence(history)
                else:
                    self._train_algo_in_nni(history)

            self.export_cache(train=True)
        else:
            logger.info("Skipping algorithm training...")

        # step 4: model ensemble and write the prediction to disks.
        if self.ensemble:
            ensemble_runner = EnsembleRunner(
                data_src_cfg_name=self.data_src_cfg_name,
                work_dir=self.work_dir,
                num_fold=self.num_fold,
                ensemble_method_name=self.ensemble_method_name,
                mgpu=int(self.device_setting["n_devices"]) > 1,
                **self.kwargs,  # for set_image_save_transform
                **self.pred_params,
            )  # for inference
            ensemble_runner.run(self.device_setting)
        logger.info("Auto3Dseg pipeline is completed successfully.")

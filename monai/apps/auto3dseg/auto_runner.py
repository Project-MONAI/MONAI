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

import os
import shutil
import subprocess
from copy import deepcopy
from time import sleep
from typing import Any, Dict, List, Optional, Union

import torch

from monai.apps.auto3dseg.bundle_gen import BundleGen
from monai.apps.auto3dseg.data_analyzer import DataAnalyzer
from monai.apps.auto3dseg.ensemble_builder import (
    AlgoEnsemble,
    AlgoEnsembleBestByFold,
    AlgoEnsembleBestN,
    AlgoEnsembleBuilder,
)
from monai.apps.auto3dseg.hpo_gen import NNIGen
from monai.apps.auto3dseg.utils import export_bundle_algo_history, import_bundle_algo_history
from monai.apps.utils import get_logger
from monai.auto3dseg.utils import algo_to_pickle
from monai.bundle import ConfigParser
from monai.transforms import SaveImage
from monai.utils.enums import AlgoEnsembleKeys
from monai.utils.module import look_up_option, optional_import

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
        analyze: on/off switch to run DataAnalyzer and generate a datastats report. If it is set to False,
            The AutoRunner will attempt to skip datastats analysis and use cached results. If there is no such cache,
            AutoRunner will report an error and stop.
        algo_gen: on/off switch to run AlgoGen and generate templated BundleAlgos. If it is set to False,
            The AutoRunner will attempt to skip the algorithm generation and stop if there is no cache to load.
        train: on/off switch to run training and generate algorithm checkpoints. If it is set to False,
            The AutoRunner will attempt to skip the training for all algorithms. If there is zero trained algorithm
            but ``train`` is set to False, AutoRunner will stop.
        hpo: use hyperparameter optimization (HPO) in the training phase. Users can provide a list of
            hyper-parameter and a search will be performed to investigate the algorithm performances.
        hpo_backend: a string that indicates the backend of the HPO. Currently, only NNI Grid-search mode
            is supported
        ensemble: on/off switch to run model ensemble and use the ensemble to predict outputs in testing
            datasets.
        not_use_cache: if the value is True, it will ignore all cached results in data analysis,
            algorithm generation, or training, and start the pipeline from scratch.
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
            input = "path_to_yaml_data_cfg"
            runner = AutoRunner(work_dir=work_dir, input=input)
            runner.run()

        - User can specify training parameters by:

        .. code-block:: python

            input = "path_to_yaml_data_cfg"
            runner = AutoRunner(input=input)
            train_param = {
                "CUDA_VISIBLE_DEVICES": [0],
                "num_iterations": 8,
                "num_iterations_per_validation": 4,
                "num_images_per_batch": 2,
                "num_epochs": 2,
            }
            runner.set_training_params(params=train_param)  # 2 epochs
            runner.run()

        - User can specify the fold number of cross validation

        .. code-block:: python

            input = "path_to_yaml_data_cfg"
            runner = AutoRunner(input=input)
            runner.set_num_fold(n_fold = 2)
            runner.run()

        - User can specify the prediction parameters during algo ensemble inference:

        .. code-block:: python

            input = "path_to_yaml_data_cfg"
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

            input = "path_to_yaml_data_cfg"
            pred_param = {
                "CUDA_VISIBLE_DEVICES": [0],
                "num_iterations": 8,
                "num_iterations_per_validation": 4,
                "num_images_per_batch": 2,
                "num_epochs": 2,
            }
            runner = AutoRunner(input=input, hpo=True)
            runner.set_nni_search_space({"learning_rate": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]}})
            runner.run()

    Notes:
        Expected results in the work_dir as below.
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

    def __init__(
        self,
        work_dir: str = "./work_dir",
        input: Union[Dict[str, Any], str, None] = None,
        analyze: bool = True,
        algo_gen: bool = True,
        train: bool = True,
        hpo: bool = False,
        hpo_backend: str = "nni",
        ensemble: bool = True,
        not_use_cache: bool = False,
        **kwargs,
    ):
        if not os.path.isdir(work_dir):
            logger.info(f"{work_dir} does not exists. Creating...")
            os.makedirs(work_dir)
            logger.info(f"{work_dir} created to save all results")
        else:
            logger.info(f"Work directory {work_dir} is used to save all results")

        self.work_dir = os.path.abspath(work_dir)
        self.data_src_cfg_name = os.path.join(self.work_dir, "input.yaml")

        if input is None:
            input = os.path.join(self.work_dir, "input.yaml")
        elif isinstance(input, Dict):
            ConfigParser.export_config_file(input, self.data_src_cfg_name)
            self.data_src_cfg = input
        elif isinstance(input, str) and os.path.isfile(input):
            shutil.copy(input, self.data_src_cfg_name)
            logger.info(f"Loading {input} for AutoRunner and making a copy in {self.data_src_cfg_name}")
            self.data_src_cfg = ConfigParser.load_config_file(self.data_src_cfg_name)
        else:
            raise ValueError(f"{input} is not a valid file")

        self.not_use_cache = not_use_cache
        self.cache_filename = os.path.join(self.work_dir, "cache.yaml")
        self.cache = self.check_cache()
        self.export_cache()

        # Whether we need all the steps or not
        self.analyze = self.check_analyze(analyze)
        self.algo_gen = self.check_algo_gen(algo_gen)
        self.train = self.check_train(train)
        self.ensemble = ensemble  # last step, no need to check

        # intermediate variables
        self.dataroot = self.data_src_cfg["dataroot"]
        self.datalist_filename = self.data_src_cfg["datalist"]
        self.datastats_filename = os.path.join(self.work_dir, "datastats.yaml")
        self.set_training_params()
        self.set_num_fold()
        self.set_prediction_params()
        self.save_image = self.set_image_save_transform(kwargs)
        self.ensemble_method: AlgoEnsemble
        self.set_ensemble_method()

        # hpo
        if hpo_backend.lower() != "nni":
            raise NotImplementedError("HPOGen backend only supports NNI")
        self.hpo = hpo and has_nni
        self.set_hpo_params()
        self.search_space: Dict[str, Dict[str, Any]] = {}
        self.hpo_tasks = 0

    def check_cache(self):
        """
        Check if the intermediate result is cached after each step in the current working directory

        Returns:
            a dict of cache results. If not_use_cache is set to True, or there is no cache file in the
            working directory, the result will be ``empty_cache`` in which all ``has_cache`` keys are
            set to False.
        """
        empty_cache = {
            "analyze": {"has_cache": False, "datastats": None},
            "algo_gen": {"has_cache": False},
            "train": {"has_cache": False},
        }
        if self.not_use_cache or not os.path.isfile(self.cache_filename):
            return empty_cache

        cache = ConfigParser.load_config_file(self.cache_filename)

        if cache["analyze"]["has_cache"]:
            # check if the file in the right format and exists.
            if not isinstance(cache["analyze"]["datastats"], str):
                cache["analyze"] = False
                cache["analyze"]["datastats"] = None

            if not os.path.isfile(cache["analyze"]["datastats"]):
                cache["analyze"]["has_cache"] = False

        if cache["algo_gen"]["has_cache"]:
            history = import_bundle_algo_history(self.work_dir, only_trained=False)
            if len(history) == 0:  # no saved algo_objects
                cache["algo_gen"]["has_cache"] = False

        if cache["train"]["has_cache"]:
            trained_history = import_bundle_algo_history(self.work_dir, only_trained=True)
            if len(trained_history) == 0:
                cache["train"]["has_cache"] = False

        return cache

    def export_cache(self):
        """
        Save the cache state as ``cache.yaml`` in the working directory
        """
        ConfigParser.export_config_file(self.cache, self.cache_filename, fmt="yaml", default_flow_style=None)

    def check_analyze(self, analyze: bool):
        """Check if the AutoRunner can skip data analysis."""

        if self.cache["analyze"]["has_cache"]:
            return False  # we can use cached result

        if analyze:
            return True  # we need to do analysis
        else:
            raise ValueError(
                f"cache data analysis report is not found in {self.work_dir}"
                "or the cache.yaml file is missing in the directory"
            )

    def check_algo_gen(self, algo_gen: bool):
        """Check if the AutoRunner can skip AlgoGen/BundleGen."""

        if self.cache["algo_gen"]["has_cache"]:
            return False  # we can use cached result

        if algo_gen:
            return True  # we need to do algo_gen
        else:
            raise ValueError(
                f"algo_object.pkl is not found in the task folders under {self.work_dir}"
                "or the cache.yaml file is missing in the directory"
            )

    def check_train(self, train: bool):
        """Check if the AutoRunner can skip training."""

        if self.cache["train"]["has_cache"]:
            return False  # we can use cached result

        if train:
            return True  # we need to do training
        else:
            raise ValueError(
                f"algo_object.pkl in the task folders under {self.work_dir} has no [best_metrics] key"
                "or the cache.yaml file is missing in the directory"
            )

    def set_num_fold(self, num_fold: int = 5):
        """
        Set the number of cross validation folds for all algos.

        Args:
            num_fold: a positive integer to define the number of folds.
        """
        if num_fold <= 0:
            raise ValueError(f"num_fold is expected to be an integer greater than zero. Now it gets {num_fold}")
        self.num_fold = num_fold

    def set_training_params(self, params: Optional[Dict[str, Any]] = None):
        """
        Set the training params for all algos.

        Args:
            params: a dict that defines the overriding key-value pairs during training. The overriding method
                is defined by the algo class.

        Examples:
            For BundleAlgo objects, the training parameter to shorten the training time to a few epochs can be
                {"num_iterations": 8, "num_iterations_per_validation": 4}

        """
        if params is None:
            self.train_params = {}
        else:
            self.train_params = deepcopy(params)

    def set_prediction_params(self, params: Optional[Dict[str, Any]] = None):
        """
        Set the prediction params for all algos.

        Args:
            params: a dict that defines the overriding key-value pairs during prediction. The overriding method
            is defined by the algo class.

        Examples:
            For BundleAlgo objects, this set of param will specify the algo ensemble to only inference the first
                two files in the testing datalist {"file_slices": slice(0, 2)}

        """
        if params is None:
            self.pred_params = {"sigmoid": True}  # output will be 0-1
        else:
            self.pred_params = deepcopy(params)

    def set_hpo_params(self, params: Optional[Dict[str, Any]] = None):
        """
        Set parameters for the HPO module and the algos before the training. It will attempt to (1) override bundle
        templates with the key-value pairs in ``params`` (2) chagne the config of the HPO module (e.g. NNI) if the
        key is found to be one of:
            - "trialCodeDirectory"
            - "trialGpuNumber"
            - "trialConcurrency"
            - "maxTrialNumber"
            - "maxExperimentDuration"
            - "tuner"
            - "trainingService"

        Args:
            params: a dict that defines the overriding key-value pairs during instantiation of the algo. For
                BundleAlgo, it will overide the template config filling.
        """
        if params is None:
            self.hpo_params = self.train_params
        else:
            self.hpo_params = params

    def set_nni_search_space(self, search_space):
        """
        Set the search space for NNI parameter search.

        Args:
            search_space: hyper paramter search space in the form of dict. For more information, please check
                NNI documentation: https://nni.readthedocs.io/en/v2.2/Tutorial/SearchSpaceSpec.html .
        """
        value_combinations = 1
        for k, v in search_space.items():
            if "_value" not in v:
                raise ValueError(f"{search_space} key {k} value {v} has not _value")
            value_combinations *= len(v["_value"])

        self.search_space = search_space
        self.hpo_tasks = value_combinations

    def set_image_save_transform(self, kwargs):
        """
        Set the ensemble output transform.

        Args:
            kwargs: image writing parameters for the ensemble inference. The kwargs format follows SaveImage
            transform. For more information, check https://docs.monai.io/en/stable/transforms.html#saveimage .

        """

        if "output_dir" in kwargs:
            output_dir = kwargs.pop("output_dir")
        else:
            output_dir = os.path.join(self.work_dir, "ensemble_output")
            logger.info(f"The output_dir is not specified. {output_dir} will be used to save ensemble predictions")

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Directory {output_dir} is created to save ensemble predictions")

        if "output_postfix" in kwargs:
            output_postfix = kwargs.pop("output_postfix")
        else:
            output_postfix = "ensemble"

        self.output_dir = output_dir
        return SaveImage(output_dir=output_dir, output_postfix=output_postfix, **kwargs)

    def set_ensemble_method(self, ensemble_method_name: str = "AlgoEnsembleBestN", **kwargs):
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
            n_best = kwargs.pop("n_best", False)
            n_best = 2 if not n_best else n_best
            self.ensemble_method = AlgoEnsembleBestN(n_best=n_best)
        elif self.ensemble_method_name == "AlgoEnsembleBestByFold":
            self.ensemble_method = AlgoEnsembleBestByFold(n_fold=self.num_fold)
        else:
            raise NotImplementedError(f"Ensemble method {self.ensemble_method_name} is not implemented.")

    def _train_algo_in_sequence(self, history: List[Dict[str, Any]]):
        """
        Train the Algos in a seqential scheme. The order of training is randomized.

        Args:
            history: the history of generated Algos. It is a list of dicts. Each element has the task name
                (e.g. "dints_0" for dints network in fold 0) as the key and the algo object as the value.
                After the training, the algo object with the ``best_metric`` will be saved as a pickle file.

        Note:
            The final results of the model training will be written to all the generated algorithm's output
                folders under the working directory. The results include the model checkpoints, a
                progress.yaml, accuracies in CSV and a pickle file of the Algo object.
        """
        for task in history:
            for _, algo in task.items():
                algo.train(self.train_params)
                acc = algo.get_score()
                algo_to_pickle(algo, template_path=algo.template_path, best_metrics=acc)

    def _train_algo_in_nni(self, history):
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
        for task in history:
            for name, algo in task.items():
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
                nni_config_filename = os.path.abspath(os.path.join(self.work_dir, "nni_config.yaml"))
                ConfigParser.export_config_file(nni_config, nni_config_filename, fmt="yaml", default_flow_style=None)

                max_trial = min(self.hpo_tasks, default_nni_config["maxTrialNumber"])
                cmd = "nnictl create --config " + nni_config_filename + " --port 8088"
                subprocess.run(cmd.split(), check=True)

                n_trainings = len(import_bundle_algo_history(self.work_dir, only_trained=True))
                while n_trainings - last_total_tasks < max_trial:
                    sleep(1)
                    n_trainings = len(import_bundle_algo_history(self.work_dir, only_trained=True))

                cmd = "nnictl stop --all"
                subprocess.run(cmd.split(), check=True)
                logger.info(f"NNI completes HPO on {name}")
                last_total_tasks = n_trainings

    def run(self):
        """
        Run the AutoRunner pipeline
        """
        # step 1: data analysis
        if self.analyze:
            da = DataAnalyzer(self.datalist_filename, self.dataroot, output_path=self.datastats_filename)
            da.get_all_case_stats()
            self.cache["analyze"]["has_cache"] = True
            self.cache["analyze"]["datastats"] = self.datastats_filename
            self.export_cache()
        else:
            logger.info("Found cached results and skipping data analysis...")

        # step 2: algorithm generation
        if self.algo_gen:
            bundle_generator = BundleGen(
                algo_path=self.work_dir,
                data_stats_filename=self.datastats_filename,
                data_src_cfg_name=self.data_src_cfg_name,
            )

            bundle_generator.generate(self.work_dir, num_fold=self.num_fold)
            history = bundle_generator.get_history()
            export_bundle_algo_history(history)
            self.cache["algo_gen"]["has_cache"] = True
            self.export_cache()
        else:
            logger.info("Found cached results and skipping algorithm generation...")

        # step 3: algo training
        if self.train:
            history = import_bundle_algo_history(self.work_dir, only_trained=False)
            if not self.hpo:
                self._train_algo_in_sequence(history)
            else:
                self._train_algo_in_nni(history)
            self.cache["train"]["has_cache"] = True
            self.export_cache()
        else:
            logger.info("Found cached results and skipping algorithm training...")

        # step 4: model ensemble and write the prediction to disks.
        if self.ensemble:
            history = import_bundle_algo_history(self.work_dir, only_trained=True)
            builder = AlgoEnsembleBuilder(history, self.data_src_cfg_name)
            builder.set_ensemble_method(self.ensemble_method)
            ensembler = builder.get_ensemble()
            preds = ensembler(pred_param=self.pred_params)  # apply sigmoid to binarize the prediction
            print("Auto3Dseg picked the following networks to ensemble:")
            for algo in ensembler.get_algo_ensemble():
                print(algo[AlgoEnsembleKeys.ID])

            for pred in preds:
                self.save_image(pred)
            logger.info(f"Auto3Dseg ensemble prediction outputs are saved in {self.output_dir}.")

        logger.info("Auto3Dseg pipeline is complete successfully.")

import subprocess
from copy import deepcopy
import os

import torch
import shutil

from monai.apps.auto3dseg.data_analyzer import DataAnalyzer
from monai.apps.auto3dseg.bundle_gen import BundleGen
from monai.apps.auto3dseg.hpo_gen import NNIGen
from monai.apps.auto3dseg.ensemble_builder import AlgoEnsembleBestByFold, AlgoEnsembleBuilder, AlgoEnsembleBestN
from monai.apps.auto3dseg.utils import import_bundle_algo_history, export_bundle_algo_history
from monai.apps.utils import get_logger
from monai.auto3dseg.utils import algo_to_pickle
from monai.transforms import SaveImage
from monai.bundle import ConfigParser
from monai.utils.enums import AlgoEnsembleKeys
from monai.utils.module import look_up_option, optional_import

from typing import Optional
from time import sleep

logger = get_logger(module_name=__name__)

nni, has_nni = optional_import("nni")

class AutoRunner:
    """
    Auto3Dseg interface for minimal usage

    Args:
        input: path to a configuration file (yaml) that contains datalist, dataroot, and other params.
                The config will be in a form of {"modality": "ct", "datalist": "path_to_json_datalist", "dataroot":
                "path_dir_data"}
        work_dir: working directory to save the intermediate results
        analyze: on/off switch for data analyzer
        algo_gen: on/off switch for algoGen
        train: on/off switch for sequential schedule of model training
        no_cache: if no_cache is True, it will reset the status and not use any previous results

    Examples:

        ..code-block:: python
            work_dir = "./work_dir"
            filename = "path_to_data_cfg"
            runner = AutoRunner(data_src_cfg_filename, work_dir)
            runner.run()

        ..code-block:: python
            work_dir = "./work_dir"
            filename = "path_to_data_cfg"
            runner = AutoRunner(data_src_cfg_filename, work_dir)
            train_param = {
                "CUDA_VISIBLE_DEVICES": [0],
                "num_iterations": 8,
                "num_iterations_per_validation": 4,
                "num_images_per_batch": 2,
                "num_epochs": 2,
            }
            runner.set_training_params(train_param)  # 2 epochs
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
        work_dir: str = './work_dir',
        input: Optional[str] = None,
        analyze: bool = True,
        algo_gen: bool = True,
        train: bool = True,
        hpo: bool = False,
        hpo_backend: str = 'nni',
        hpo_search_space = {},
        ensemble: bool = True,
        not_use_cache: bool = False,
        **kwargs
    ):  
        if not os.path.isdir(work_dir):
            logger.info(f"{work_dir} does not exists. Creating...")
            os.makedirs(work_dir)
            logger.info(f"{work_dir} created to save all results")
        else:
            logger.info(f"Work directory {work_dir} is used to save all results")

        self.work_dir = os.path.abspath(work_dir)

        if input is None:
            input = os.path.join(self.work_dir, 'input.yaml')
        else:
            if not os.path.isfile(input):
                raise ValueError(f"{input} is not a valid file")
        
        self.data_src_cfg_name = os.path.join(self.work_dir, 'input.yaml')
        shutil.copy(input, self.data_src_cfg_name)
        logger.info(f"Loading {input} for AutoRunner and making a copy in {self.data_src_cfg_name}")
        self.data_src_cfg = ConfigParser.load_config_file(self.data_src_cfg_name)

        self.not_use_cache = not_use_cache
        self.cache_filename = os.path.join(self.work_dir, 'cache.yaml')
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
        self.datastats_filename = os.path.join(self.work_dir, 'datastats.yaml')
        self.set_training_params()
        self.set_num_fold()
        self.set_prediction_params()
        self.save_image = self.set_image_save_transform(kwargs)
        self.set_ensemble_method()
        
        # hpo 
        if hpo_backend.lower() != "nni":
            raise NotImplementedError("HPOGen backend only supports NNI")
        self.hpo = hpo and has_nni
        self.set_hpo_params()
        self.search_space = {}
        self.hpo_tasks = 0

    def check_cache(self):
        """
        Check if there is cached results for each step in the current working directory

        Returns:
            a dict of cache results. The result will be ``empty_cache`` whose all ``has_cache`` keys are False,
                if not_use_cache is set to True, or ther is no cache file in the directory
        """
        empty_cache = {
            "analyze": {"has_cache": False, "datastats_file": None},
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
        Save the cache.yaml file in the working directory
        """
        ConfigParser.export_config_file(self.cache, self.cache_filename, fmt='yaml', default_flow_style=None)

    def check_analyze(self, analyze):
        """Set the AutoRunner to run DataAnalyzer. """

        if self.cache["analyze"]["has_cache"]:
            return False  # we can use cached result

        if analyze:
            return True   # we need to do analysis
        else:
            raise ValueError(f"cache data analysis report is not found in {self.work_dir}"
                "or the cache.yaml file is missing in the directory")

    def check_algo_gen(self, algo_gen):
        """Set the AutoRunner to run AlgoGen/BundleGen. """

        if self.cache["algo_gen"]["has_cache"]:
            return False  # we can use cached result

        if algo_gen:
            return True   # we need to do algo_gen
        else:
            raise ValueError(f"algo_object.pkl is not found in the task folders under {self.work_dir}"
                "or the cache.yaml file is missing in the directory")

    def check_train(self, train):
        """Set the AutoRunner to train network. """

        if self.cache["train"]["has_cache"]:
            return False  # we can use cached result

        if train:
            return True   # we need to do training
        else:
            raise ValueError(f"algo_object.pkl in the task folders under {self.work_dir} has no [best_metrics] key"
                "or the cache.yaml file is missing in the directory")

    def set_num_fold(self, num_fold: int = 5):
        """set number of cross validation folds"""
        if num_fold <= 0:
            raise ValueError(f"num_fold is expected to be an integer greater than zero. Now it gets {num_fold}")
        self.num_fold = num_fold

    def set_training_params(self, params=None):
        if params is None:
            gpus = [_i for _i in range(torch.cuda.device_count())]
            self.train_params = {"CUDA_VISIBLE_DEVICES": gpus}
        else:
            self.train_params = params

    def set_prediction_params(self, params=None):
        if params is None:
            self.pred_params = {"sigmoid": True}
        else:
            self.pred_params = params

    def set_hpo_params(self, params=None):
        if params is None:
            self.hpo_params = self.train_params
        else:
            self.hpo_params = params

    def set_hpo_search_space(self, search_space):
        
        value_combinations = 1
        for k, v in search_space.items():
            if '_value' not in v:
                raise ValueError(f"{search_space} key {k} value {v} has not _value")
            value_combinations *= len(v['_value'])
        
        self.search_space = search_space
        self.hpo_tasks = value_combinations

    def set_image_save_transform(self, kwargs):
        """How the ensemble output will be saved (formats/output_dir)"""

        if "output_dir" in kwargs:
            output_dir = kwargs.pop("output_dir")
        else:
            output_dir = os.path.join(self.work_dir, 'ensemble_output')
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
    
    def set_ensemble_method(self, mode: str = 'AlgoEnsembleBestN', **kwargs):
        """"""
        self.ensemble_mode = look_up_option(mode, supported=['AlgoEnsembleBestN', 'AlgoEnsembleBestByFold'])
        if self.ensemble_mode == 'AlgoEnsembleBestN':
            n_best = kwargs.pop('n_best', False)
            n_best = 2 if not n_best else n_best 
            self.ensemble_method = AlgoEnsembleBestN(n_best=n_best)
        elif self.ensemble_mode == 'AlgoEnsembleBestByFold':
            self.ensemble_method = AlgoEnsembleBestByFold(n_fold=self.num_fold)
        else:
            raise NotImplementedError(f"Ensemble method {self.ensemble_mode} is not implemented or registerred.")
    
    def train_algo_in_sequence(self, history):
        for task in history:
            for _, algo in task.items():
                algo.train(self.train_params)
                acc = algo.get_score()
                algo_to_pickle(algo, template_path=algo.template_path, best_metrics=acc)

    def train_algo_in_hpo(self, history):
        default_nni_config = {
            "trialCodeDirectory": "." ,
            "trialGpuNumber": torch.cuda.device_count(),
            "trialConcurrency": 1,
            "maxTrialNumber": 10,
            "maxExperimentDuration": "1h",
            "tuner": {"name": "GridSearch"},
            "trainingService":{"platform": "local", "useActiveGpu": True},
        }

        last_total_tasks = len(import_bundle_algo_history(self.work_dir, only_trained=True))
        for task in history:
            for name, algo in task.items():
                nni_gen = NNIGen(algo=algo, params=self.hpo_params)
                obj_filename = nni_gen.get_obj_filename()
                nni_config = deepcopy(default_nni_config)
                nni_config.update({"experimentName": name})
                nni_config.update({"search_space": self.search_space})
                trialCommand = "python -m monai.apps.auto3dseg NNIGen run_algo " + obj_filename + " " + self.work_dir
                nni_config.update({"trialCommand": trialCommand})
                nni_config_filename = os.path.abspath(os.path.join(self.work_dir, "nni_config.yaml"))
                ConfigParser.export_config_file(nni_config, nni_config_filename, fmt='yaml', default_flow_style=None)

                max_trial = min(self.hpo_tasks, default_nni_config["maxTrialNumber"])
                cmd = "nnictl create --config " + nni_config_filename + " --port 8088"
                subprocess.run(cmd.split(), check=True)

                n_trainings = len(import_bundle_algo_history(self.work_dir, only_trained=True))
                while(n_trainings - last_total_tasks < max_trial):
                    sleep(1)
                    n_trainings = len(import_bundle_algo_history(self.work_dir, only_trained=True))

                cmd = "nnictl stop --all"
                subprocess.run(cmd.split(), check=True)
                logger.info(f"NNI completes HPO on {name}")
                last_total_tasks = n_trainings

    def run(self):
        """
        Run the autorunner
        """
        ## data analysis
        if self.analyze:
            da = DataAnalyzer(self.datalist_filename, self.dataroot, output_path=self.datastats_filename)
            da.get_all_case_stats()
            self.cache["analyze"]["has_cache"] = True
            self.cache["analyze"]["datastats"] = self.datastats_filename
            self.export_cache()
        else:
            logger.info("Found cached results and skipping data analysis...")

        ## algorithm generation
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

        ## model training
        if self.train:
            history = import_bundle_algo_history(self.work_dir, only_trained=False)
            if not self.hpo:
                self.train_algo_in_sequence(history)
            else:
                self.train_algo_in_hpo(history)
            self.cache["train"]["has_cache"] = True
            self.export_cache()
        else:
            logger.info("Found cached results and skipping algorithm training...")

        ## model ensemble
        if self.ensemble:
            history = import_bundle_algo_history(self.work_dir, only_trained=True)
            builder = AlgoEnsembleBuilder(history, self.data_src_cfg_name)
            builder.set_ensemble_method(self.ensemble_method)
            ensembler = builder.get_ensemble()
            preds = ensembler(pred_param=self.pred_params)  # apply sigmoid to binarize the prediction
            print(f"Auto3Dseg picked the following networks to ensemble:")
            for algo in ensembler.get_algo_ensemble():
                print(algo[AlgoEnsembleKeys.ID])
                
            for pred in preds:
                self.save_image(pred)
            logger.info(f"Auto3Dseg ensemble prediction outputs are saved in {self.output_dir}.")
        
        logger.info(f"Auto3Dseg pipeline is complete successfully.")
            

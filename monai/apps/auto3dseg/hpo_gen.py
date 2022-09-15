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
from abc import abstractmethod
from copy import deepcopy
from warnings import warn

from monai.apps.auto3dseg.bundle_gen import BundleAlgo
from monai.apps.utils import get_logger
from monai.auto3dseg import Algo, AlgoGen, algo_from_pickle, algo_to_pickle
from monai.bundle.config_parser import ConfigParser
from monai.utils import optional_import

nni, has_nni = optional_import("nni")
optuna, has_optuna = optional_import("optuna")
logger = get_logger(module_name=__name__)

__all__ = ["HPOGen", "NNIGen", "OptunaGen"]


class HPOGen(AlgoGen):
    """
    The base class for hyperparameter optimization (HPO) interfaces to generate algos in the Auto3Dseg pipeline.
    The auto-generated algos are saved at their ``output_path`` on the disk. The files in the ``output_path``
    may contain scripts that define the algo, configuration files, and pickle files that save the internal states
    of the algo before/after the training. Compared to the BundleGen class, HPOGen generates Algo on-the-fly, so
    training and algo generation may be executed alternatively and take a long time to finish the generation process.

    """

    @abstractmethod
    def get_hyperparameters(self):
        """Get the hyperparameter from HPO."""
        raise NotImplementedError

    @abstractmethod
    def update_params(self, *args, **kwargs):
        """Update Algo parameters according to the hyperparameters to be evaluated."""
        raise NotImplementedError

    @abstractmethod
    def set_score(self):
        """Report the evaluated results to HPO."""
        raise NotImplementedError

    @abstractmethod
    def run_algo(self, *args, **kwargs):
        """Interface for launch the training given the fetched hyperparameters."""
        raise NotImplementedError


class NNIGen(HPOGen):
    """
    Generate algorithms for the NNI to automate hyperparameter tuning. The module has two major interfaces:
    ``__init__`` which prints out how to set up the NNI, and a trialCommand function ``run_algo`` for the NNI library to
    start the trial of the algo. More about trialCommand function can be found in ``trail code`` section in NNI webpage
    https://nni.readthedocs.io/en/latest/tutorials/hpo_quickstart_pytorch/main.html .

    Args:
        algo: an Algo object (e.g. BundleAlgo) with defined methods: ``get_output_path`` and train
            and supports saving to and loading from pickle files via ``algo_from_pickle`` and ``algo_to_pickle``.
        params: a set of parameter to override the algo if override is supported by Algo subclass.

    Examples::

        The experiment will keep generating new folders to save the model checkpoints, scripts, and configs if available.
        ├── algorithm_templates
        │   └── unet
        ├── unet_0
        │   ├── algo_object.pkl
        │   ├── configs
        │   └── scripts
        ├── unet_0_learning_rate_0.01
        │   ├── algo_object.pkl
        │   ├── configs
        │   ├── model_fold0
        │   └── scripts
        └── unet_0_learning_rate_0.1
            ├── algo_object.pkl
            ├── configs
            ├── model_fold0
            └── scripts

    Notes:
        The NNIGen will prepare the algorithms in a folder and suggest a command to replace trialCommand in the experiment
        config. However, NNIGen will not trigger NNI. User needs to write their NNI experiment configs, and then run the
        NNI command manually.
    """

    def __init__(self, algo=None, params=None):
        self.algo: Algo
        self.hint = ""
        self.obj_filename = ""

        if algo is not None:
            if isinstance(algo, BundleAlgo):
                if params is None:
                    self.algo = algo
                else:
                    self.algo = deepcopy(algo)
                    name = os.path.basename(algo.get_output_path()) + "_override"
                    output_folder = os.path.dirname(algo.get_output_path())

                    params.update({"fill_with_datastats": False})  # just copy, not using datastats to fill
                    self.algo.export_to_disk(output_folder, name, **params)
            else:
                self.algo = algo

            if isinstance(algo, BundleAlgo):
                self.obj_filename = algo_to_pickle(self.algo, template_path=self.algo.template_path)
                self.print_bundle_algo_instruction()
            else:
                self.obj_filename = algo_to_pickle(self.algo)
                # nni instruction unknown

    def get_obj_filename(self):
        """Return the filename of the dumped pickle algo object."""
        return self.obj_filename

    def print_bundle_algo_instruction(self):
        """
        Print how to write the trial commands for Bundle Algo.
        """
        hint = "python -m monai.apps.auto3dseg NNIGen run_algo "
        logger.info("=" * 140)
        logger.info("If NNI will run in your local env: ")
        logger.info("1. Add the following line to the trialCommand in your NNI config: ")
        logger.info(f"{hint} {self.obj_filename} {{result_dir}}")
        logger.info("-" * 140)
        logger.info("If NNI will run in a remote env: ")
        logger.info(
            f"1. Copy the algorithm_templates folder {self.algo.template_path} to remote {{remote_algorithm_templates_dir}}"
        )
        logger.info(f"2. Copy the older {self.algo.get_output_path()} to the remote machine {{remote_algo_dir}}")
        logger.info("Then add the following line to the trialCommand in your NNI config: ")
        logger.info(f"{hint} {{remote_algo_dir}} {{result_dir}} {{remote_algorithm_templates_dir}}")
        logger.info("=" * 140)

    def get_hyperparameters(self):
        """
        Get parameter for next round of training from NNI server.
        """
        if has_nni:
            return nni.get_next_parameter()
        warn("NNI is not detected. The code will continue to run without NNI.")
        return {}

    def update_params(self, params: dict):  # type: ignore
        """
        Translate the parameter from monai bundle to meet NNI requirements.

        Args:
            params: a dict of parameters.
        """
        self.params = params

    def get_task_id(self):
        """
        Get the identifier of the current experiment. In the format of listing the searching parameter name and values
        connected by underscore in the file name.
        """
        return "".join(f"_{k}_{v}" for k, v in self.params.items()) or "_None"

    def generate(self, output_folder: str = ".") -> None:
        """
        Generate the record for each Algo. If it is a BundleAlgo, it will generate the config files.

        Args:
            output_folder: the directory nni will save the results to.
        """
        task_id = self.get_task_id()
        task_prefix = os.path.basename(self.algo.get_output_path())
        write_path = os.path.join(output_folder, task_prefix + task_id)
        self.obj_filename = os.path.join(write_path, "algo_object.pkl")

        if isinstance(self.algo, BundleAlgo):
            self.algo.export_to_disk(output_folder, task_prefix + task_id, fill_with_datastats=False)
        else:

            ConfigParser.export_config_file(self.params, write_path)
            logger.info(write_path)

    def set_score(self, acc):
        """
        Report the acc to NNI server.
        """
        if has_nni:
            nni.report_final_result(acc)
        else:
            warn("NNI is not detected. The code will continue to run without NNI.")

    def run_algo(self, obj_filename: str, output_folder: str = ".", template_path=None) -> None:  # type: ignore
        """
        The python interface for NNI to run.

        Args:
            obj_filename: the pickle-exported Algo object.
            output_folder: the root path of the algorithms templates.
            template_path: the algorithm_template. It must contain algo.py in the follow path:
                ``{algorithm_templates_dir}/{network}/scripts/algo.py``
        """
        if not os.path.isfile(obj_filename):
            raise ValueError(f"{obj_filename} is not found")

        self.algo, algo_meta_data = algo_from_pickle(obj_filename, template_path=template_path)

        if isinstance(self.algo, BundleAlgo):  # algo's template path needs override
            self.algo.template_path = algo_meta_data["template_path"]

        # step 1 sample hyperparams
        params = self.get_hyperparameters()
        # step 2 set the update params for the algo to run in the next trial
        self.update_params(params)
        # step 3 generate the folder to save checkpoints and train
        self.generate(output_folder)
        self.algo.train(self.params)
        # step 4 report validation acc to controller
        acc = self.algo.get_score()
        if isinstance(self.algo, BundleAlgo):
            algo_to_pickle(self.algo, template_path=self.algo.template_path, best_metrics=acc)
        else:
            algo_to_pickle(self.algo, best_metrics=acc)
        self.set_score(acc)


class OptunaGen(HPOGen):
    """
    Generate algorithms for the Optuna to automate hyperparameter tuning. Please refer to NNI and Optuna
    (https://optuna.readthedocs.io/en/stable/) for more information. Optuna has different running scheme
    compared to NNI. The hyperparameter samples come from a trial object (trial.suggest...) created by Optuna,
    so OptunaGen needs to accept this trial object as input. Meanwhile, Optuna calls OptunaGen,
    thus OptunaGen.__call__() should return the accuracy. Use functools.partial to wrap OptunaGen
    for addition input arguments.

    Args:
        algo: an Algo object (e.g. BundleAlgo). The object must at least define two methods: get_output_path and train
            and supports saving to and loading from pickle files via ``algo_from_pickle`` and ``algo_to_pickle``.
        params: a set of parameter to override the algo if override is supported by Algo subclass.

    Examples::

        The experiment will keep generating new folders to save the model checkpoints, scripts, and configs if available.
        ├── algorithm_templates
        │   └── unet
        ├── unet_0
        │   ├── algo_object.pkl
        │   ├── configs
        │   └── scripts
        ├── unet_0_learning_rate_0.01
        │   ├── algo_object.pkl
        │   ├── configs
        │   ├── model_fold0
        │   └── scripts
        └── unet_0_learning_rate_0.1
            ├── algo_object.pkl
            ├── configs
            ├── model_fold0
            └── scripts

    Notes:
        Different from NNI and NNIGen, OptunaGen and Optuna can be ran within the Python process.

    """

    def __init__(self, algo=None, params=None):
        self.algo: Algo
        self.obj_filename = ""

        if algo is not None:
            if isinstance(algo, BundleAlgo):
                if params is None:
                    self.algo = algo
                else:
                    self.algo = deepcopy(algo)
                    name = os.path.basename(algo.get_output_path()) + "_override"
                    output_folder = os.path.dirname(algo.get_output_path())

                    params.update({"fill_with_datastats": False})  # just copy, not using datastats to fill
                    self.algo.export_to_disk(output_folder, name, **params)
            else:
                self.algo = algo

            if isinstance(algo, BundleAlgo):
                self.obj_filename = algo_to_pickle(self.algo, template_path=self.algo.template_path)
            else:
                self.obj_filename = algo_to_pickle(self.algo)
                # nni instruction unknown

    def get_obj_filename(self):
        """Return the dumped pickle object of algo."""
        return self.obj_filename

    def get_hyperparameters(self):
        """
        Get parameter for next round of training from optuna trial object.
        This function requires user rewrite during usage for different search space.
        """
        if has_optuna:
            logger.info("Please rewrite this code by creating a child class")
            return {"learning_rate": self.trial.suggest_float("learning_rate", 0.0001, 0.1)}
        else:
            warn("Optuna is not detected. The code will continue to run without Optuna.")
            return {}

    def set_score(self, acc):
        """Set the accuracy score"""
        self.acc = acc

    def set_trial(self, trial):
        """Set the Optuna trial"""
        self.trial = trial

    def __call__(self, trial, obj_filename: str, output_folder: str = ".", template_path=None):
        """
        Callabe that Optuna will use to optimize the hyper-parameters

        Args:
            obj_filename: the pickle-exported Algo object.
            output_folder: the root path of the algorithms templates.
            template_path: the algorithm_template. It must contain algo.py in the follow path:
                ``{algorithm_templates_dir}/{network}/scripts/algo.py``
        """
        self.set_trial(trial)
        self.run_algo(obj_filename, output_folder, template_path)
        return self.acc

    def update_params(self, params: dict):  # type: ignore
        """
        Translate the parameter from monai bundle.

        Args:
            params: a dict of parameters.
        """
        self.params = params

    def get_task_id(self):
        """
        Get the identifier of the current experiment. In the format of listing the searching parameter name and values
        connected by underscore in the file name.
        """
        return "".join(f"_{k}_{v}" for k, v in self.params.items()) or "_None"

    def generate(self, output_folder: str = ".") -> None:
        """
        Generate the record for each Algo. If it is a BundleAlgo, it will generate the config files.

        Args:
            output_folder: the directory nni will save the results to.
        """
        task_id = self.get_task_id()
        task_prefix = os.path.basename(self.algo.get_output_path())
        write_path = os.path.join(output_folder, task_prefix + task_id)
        self.obj_filename = os.path.join(write_path, "algo_object.pkl")

        if isinstance(self.algo, BundleAlgo):
            self.algo.export_to_disk(output_folder, task_prefix + task_id, fill_with_datastats=False)
        else:

            ConfigParser.export_config_file(self.params, write_path)
            logger.info(write_path)

    def run_algo(self, obj_filename: str, output_folder: str = ".", template_path=None) -> None:  # type: ignore
        """
        The python interface for NNI to run.

        Args:
            obj_filename: the pickle-exported Algo object.
            output_folder: the root path of the algorithms templates.
            template_path: the algorithm_template. It must contain algo.py in the follow path:
                ``{algorithm_templates_dir}/{network}/scripts/algo.py``
        """
        if not os.path.isfile(obj_filename):
            raise ValueError(f"{obj_filename} is not found")

        self.algo, algo_meta_data = algo_from_pickle(obj_filename, template_path=template_path)

        if isinstance(self.algo, BundleAlgo):  # algo's template path needs override
            self.algo.template_path = algo_meta_data["template_path"]

        # step 1 sample hyperparams
        params = self.get_hyperparameters()
        # step 2 set the update params for the algo to run in the next trial
        self.update_params(params)
        # step 3 generate the folder to save checkpoints and train
        self.generate(output_folder)
        self.algo.train(self.params)
        # step 4 report validation acc to controller
        acc = self.algo.get_score()
        if isinstance(self.algo, BundleAlgo):
            algo_to_pickle(self.algo, template_path=self.algo.template_path, best_metrics=acc)
        else:
            algo_to_pickle(self.algo, best_metrics=acc)
        self.set_score(acc)

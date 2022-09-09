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
from typing import List

from monai.apps.auto3dseg import BundleAlgo
from monai.apps.utils import get_logger
from monai.auto3dseg import Algo, AlgoGen, algo_export_to_pickle, algo_import_from_pickle
from monai.bundle.config_parser import ConfigParser
from monai.utils import optional_import

nni, has_nni = optional_import("nni")

logger = get_logger(module_name=__name__)

__all__ = ["HPOGen", "NNIGen"]


class HPOGen(AlgoGen):
    """
    This class generates a set of al, each of them can run independently.
    """

    @abstractmethod
    def get_hyperparameters(self):
        """Get the hyperparameter from HPO"""
        raise NotImplementedError("")

    @abstractmethod
    def update_params(self, *args, **kwargs):  # type: ignore
        """Update model params"""
        raise NotImplementedError("")

    @abstractmethod
    def set_score(self):
        """Report result to HPO"""
        raise NotImplementedError("")

    @abstractmethod
    def run_algo(self, *args, **kwargs):  # type: ignore
        """Interface for the HPO to run the training"""
        raise NotImplementedError("")

    @abstractmethod
    def get_history(self, *args, **kwargs):  # type: ignore
        """Interface for the HPO to run the history"""
        raise NotImplementedError("")


class NNIGen(HPOGen):
    """
    Generate algorithms for the NNI to automate algorithm hyper-parameter tuning. The module has two major interfaces:
    ``__init__`` which sets up the algorithm, and a trialCommand function ``run_algo`` for the NNI library. More about
    trialCommand function can be found in ``trail code`` section in NNI webpage
    https://nni.readthedocs.io/en/latest/tutorials/hpo_quickstart_pytorch/main.html .

    Args:
        algo_dict: an dict that has {name: Algo object} format. The Algo object must define two methods: get_output_path
            and train.
        params: a set of parameter to override the algo if override is supported by Algo subclass.

    Raises:
        ValueError if the user tries to override a algo that doesn't have ``export_to_disk`` function.

    Examples:
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

    def __init__(self, algo_path: str = ".", algo=None, params=None):
        self.algo: Algo
        self.hint = ""
        self.obj_filename = ""
        self.algo_templates_dir = os.path.join(algo_path, "algorithm_templates")
        self.base_task_dir = ""

        if not os.path.isdir(self.algo_templates_dir):
            raise ValueError(f"{self.algo_templates_dir} is not a directory.")

        if algo is not None:
            self.base_task_dir = algo.get_output_path()

            if params is None:
                self.obj_filename = os.path.join(self.base_task_dir, "algo_object.pkl")
                algo_export_to_pickle(algo, self.obj_filename, self.algo_templates_dir)
            elif isinstance(algo, BundleAlgo):
                self.base_task_dir += "_override"
                task_name = os.path.basename(self.base_task_dir)
                output_folder = os.path.dirname(self.base_task_dir)
                algo_override = deepcopy(algo)  # avoid overriding the existing algo
                algo_override.export_to_disk(output_folder, task_name, **params)

                self.obj_filename = os.path.join(self.base_task_dir, "algo_object.pkl")
                algo_export_to_pickle(algo_override, self.obj_filename, self.algo_templates_dir)
            else:
                raise ValueError(f"{algo.__class__} does not support param override")

            self.print_nni_instruction()

    def get_obj_filename(self):
        """Return the dumped pickle object of algo"""
        return self.obj_filename

    def print_nni_instruction(self):
        """
        Print how to write the trial commands in NNI
        """
        hint = "python -m monai.apps.auto3dseg NNIGen run_algo "
        logger.info("=" * 140)
        logger.info("If NNI will run in your local env: ")
        logger.info("1. Add the following line to the trialCommand in your NNI config: ")
        logger.info(f"{hint} {self.obj_filename} {{result_dir}}")
        logger.info("-" * 140)
        logger.info("If NNI will run in a remote env: ")
        logger.info(
            f"1. Copy the algorithm_templates folder {self.algo_templates_dir} to remote {{remote_algorithm_templates_dir}}"
        )
        logger.info(f"2. Copy the base_task folder {self.base_task_dir} to the remote machine {{remote_base_task_dir}}")
        logger.info("Then add the following line to the trialCommand in your NNI config: ")
        logger.info(f"{hint} {{remote_base_task_dir}} {{result_dir}} {{remote_algorithm_templates_dir}}")
        logger.info("=" * 140)

    def get_hyperparameters(self):
        """
        Get parameter for next round of training from nni server
        """
        if has_nni:
            return nni.get_next_parameter()
        else:
            return {}

    def update_params(self, params: dict):  # type: ignore
        """
        Translate the parameter from monai bundle to nni format.

        Args:
            params: a dict of parameters.
        """
        self.params = params

    def get_task_id(self):
        """
        Get the identifier of the current experiment. In the format of listing the searching parameter name and values
        connected by underscore in the file name.
        """
        task_id = ""
        for k, v in self.params.items():
            task_id += f"_{k}_{v}"
        if len(task_id) == 0:
            task_id = "_None"  # avoid rewriting the original
        return task_id

    def generate(self, output_folder: str = ".") -> None:
        """
        Generate the record for each Algo. If it is a BundleAlgo, it will generate the config files.

        Args:
            output_folder
        """
        task_id = self.get_task_id()
        task_prefix = os.path.basename(self.algo.get_output_path())
        write_path = os.path.join(output_folder, task_prefix + task_id)
        self.obj_filename = os.path.join(write_path, "algo_object.pkl")

        if isinstance(self.algo, BundleAlgo):
            self.algo.export_to_disk(output_folder, task_prefix + task_id)
            algo_export_to_pickle(self.algo, self.obj_filename, self.algo_templates_dir)
        else:

            ConfigParser.export_config_file(self.params, write_path)
            logger.info(write_path)

    def set_score(self, acc):
        """
        Report the acc to nni server
        """
        if has_nni:
            nni.report_final_result(acc)
        return

    def run_algo(self, obj_filename: str, output_folder: str = ".", algo_templates_dir=None) -> None:  # type: ignore
        """
        The python interface for NNI to run

        Args:
            obj_filename: the pickle-exported Algo object.
            output_folder: the root path of the algorithms templates.
            algo_templates_dir: the algorithm_template. It must contain algo.py in the follow path:
                ``{algorithm_templates_dir}/{network}/scripts/algo.py``
        """
        if not os.path.isfile(obj_filename):
            raise ValueError(f"{obj_filename} is not found")

        self.algo, self.algo_templates_dir = algo_import_from_pickle(obj_filename, algo_templates_dir)

        # step1 sample hyperparams
        params = self.get_hyperparameters()
        # step 2 update model
        self.update_params(params)
        # step 3 generate the folder to save checkpoints and train
        self.generate(output_folder)
        self.algo.train(self.params)
        # step 4 report validation acc to controller
        acc = self.algo.get_score()
        self.set_score(acc)

    def get_history(self, output_folder: str = ".", algo_templates_dir=None) -> List:  # type: ignore
        """
        Get the history of the bundleAlgo object with their names/identifiers

        Args:
            output_folder: the root path of the algorithms templates.
            algo_templates_dir: the algorithm_template. It must contain algo.py in the follow path:
                ``{algorithm_templates_dir}/{network}/scripts/algo.py``
        """

        history = []

        for task_name in os.listdir(output_folder):
            write_path = os.path.join(output_folder, task_name)

            if not os.path.isdir(write_path):
                continue

            obj_filename = os.path.join(write_path, "algo_object.pkl")
            if not os.path.isfile(obj_filename):  # saved mode pkl
                continue

            algo, _ = algo_import_from_pickle(obj_filename, algo_templates_dir)
            history.append({task_name: algo})

        return history

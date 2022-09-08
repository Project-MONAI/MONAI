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
import pickle
import sys
from abc import abstractmethod
from copy import deepcopy

from monai.apps.utils import get_logger
from monai.auto3dseg.algo_gen import AlgoGen
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
    def get_hyperparameters():
        """Get the hyperparameter from HPO"""
        raise NotImplementedError("")

    @abstractmethod
    def update_params():
        """Update model params"""
        raise NotImplementedError("")

    @abstractmethod
    def set_score():
        """Report result to HPO"""
        raise NotImplementedError("")

    @abstractmethod
    def run_algo(self, obj_file, template_path):
        """Interface for the HPO to run the training"""
        raise NotImplementedError("")


class NNIGen(HPOGen):
    """
    Generate algorithms for the NNI to automate algorithm hyper parameter tuning. The module has two major functions:
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
            ├── unet_0
            │   └── unet
            │       ├── configs
            │       └── scripts
            └── unet_0_learning_rate_0.0001
                ├── model_fold0
                │   ├── accuracy_history.csv
                │   ├── best_metric_model.pt
                │   ├── Events
                │   └── progress.yaml
                └── unet
                    ├── configs
                    └── scripts

    Notes:
        The NNIGen will prepare the algorithms in a folder and suggest a command to replace trialCommand in the experiment
        config. However, NNIGen will not trigger NNI. User needs to write their NNI experiment configs, and then run the
        NNI command manually.
    """

    def __init__(self, algo_dict=None, params=None):
        self.algo = None
        self.task_prefix = None

        if algo_dict is not None:
            if len(algo_dict.keys()) > 1:
                raise ValueError(f"object {algo_dict} only allows 1 key")
            name = list(algo_dict.keys())[0]  # the only key is the name of the model

            algo = algo_dict[name]

            base_task_dir = algo.get_output_path()
            if params is None:
                obj_bytes = pickle.dumps(algo)
            elif hasattr(algo, "export_to_disk") and callable(getattr(algo, "export_to_disk")):
                base_task_dir += "_override"
                task_name = os.path.basename(base_task_dir)
                output_folder = os.path.dirname(base_task_dir)
                algo_override = deepcopy(algo)  # avoid overriding the existing algo
                algo_override.export_to_disk(output_folder, task_name, **params)
                obj_bytes = pickle.dumps(algo_override)
            else:
                raise ValueError(f"{algo.__class__} does not support param override")

            obj_file = os.path.join(base_task_dir, "algo_object.pkl")
            with open(obj_file, "wb") as f_pi:
                f_pi.write(obj_bytes)

            logger.info("Add the following line in the trialCommand in your NNI config: ")
            logger.info(f"python -m monai.apps.auto3dseg NNIGen run_algo {base_task_dir} folder/to/hpo/results/")

    def get_hyperparameters(self) -> dict:
        """
        Get parameter for next round of training from nni server
        """
        if has_nni:
            return nni.get_next_parameter()
        else:
            return {}

    def update_params(self, params):  # generate
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

    def generate(self, output_folder="."):
        """
        Generate the record for each Algo. If it is a BundleAlgo, it will generate the config files.

        Args:
            output_folder
        """
        task_id = self.get_task_id()
        if hasattr(self.algo, "export_to_disk") and callable(getattr(self.algo, "export_to_disk")):
            self.algo.export_to_disk(output_folder, self.task_prefix + task_id)
        else:
            write_path = os.path.join(output_folder, self.task_prefix + task_id)
            ConfigParser.export_config_file(self.params, write_path)
            logger.info(write_path)

    def set_score(self, acc):
        """
        Report the acc to nni server
        """
        if has_nni:
            nni.report_final_result(acc)
        return

    def run_algo(self, base_task_dir, output_folder="."):
        """
        The python interface for NNI to run

        Args:
            base_task_dir:
            output_folder: the root path of the algorithms templates.

        ..code-block:: python
            python -m monai.apps.auto3dseg NNIGen run_algo "algo.pkl" "template_dir"  #in nni
        """
        if not os.path.isdir(base_task_dir):
            raise ValueError(f"{base_task_dir} is not a directory")

        self.task_prefix = os.path.basename(base_task_dir)
        obj_file = os.path.join(base_task_dir, "algo_object.pkl")
        if not os.path.isfile(obj_file):
            raise ValueError(f"{obj_file} is not found in {base_task_dir}")

        sys.path.insert(0, base_task_dir)

        with open(obj_file, "rb") as f_pi:
            algo_bytes = f_pi.read()
        self.algo = pickle.loads(algo_bytes)

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

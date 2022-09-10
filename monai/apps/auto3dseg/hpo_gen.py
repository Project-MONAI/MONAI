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

from monai.apps.auto3dseg import BundleAlgo
from monai.apps.utils import get_logger
from monai.auto3dseg import Algo, AlgoGen, algo_from_pickle, algo_to_pickle
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
        """Get the hyperparameter from HPO."""
        raise NotImplementedError

    @abstractmethod
    def update_params(self, *args, **kwargs):  # type: ignore
        """Update model params."""
        raise NotImplementedError

    @abstractmethod
    def set_score(self):
        """Report result to HPO."""
        raise NotImplementedError

    @abstractmethod
    def run_algo(self, *args, **kwargs):  # type: ignore
        """Interface for the HPO to run the training."""
        raise NotImplementedError


class NNIGen(HPOGen):
    """
    Generate algorithms for the NNI to automate algorithm hyper-parameter tuning. The module has two major interfaces:
    ``__init__`` which sets up the algorithm, and a trialCommand function ``run_algo`` for the NNI library. More about
    trialCommand function can be found in ``trail code`` section in NNI webpage
    https://nni.readthedocs.io/en/latest/tutorials/hpo_quickstart_pytorch/main.html .

    Args:
        algo: an Algo object (e.g. BundleAlgo). The object must at least define two methods: get_output_path and train
            and supports saving to and loading from pickle files via ``algo_from_pickle`` and ``algo_to_pickle``.
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
                self.print_bundle_algo_nni_instruction()
            else:
                self.obj_filename = algo_to_pickle(self.algo)
                # nni instruction unknown

    def get_obj_filename(self):
        """Return the dumped pickle object of algo."""
        return self.obj_filename

    def print_bundle_algo_nni_instruction(self):
        """
        Print how to write the trial commands in NNI.
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
        Get parameter for next round of training from nni server.
        """
        if has_nni:
            return nni.get_next_parameter()
        else:
            warn("NNI is not detected. The code will continue to run without NNI.")
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
        Report the acc to nni server.
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

        # step1 sample hyperparams
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
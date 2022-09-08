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

from multiprocessing.sharedctypes import Value
import os
import pickle
import sys
from abc import abstractmethod
from copy import deepcopy

import nni

from monai.apps.utils import get_logger
from monai.auto3dseg.algo_gen import AlgoGen
from monai.bundle.config_parser import ConfigParser

logger = get_logger(module_name=__name__)

__all__ = ["HpoWrap", "NniWrapper"]


class HpoWrap(AlgoGen):
    """
    This class generates a set of al, each of them can run independently.
    """

    @abstractmethod
    def _get_hyperparameters():
        """Get the hyperparameter from HPO"""
        raise NotImplementedError("")

    @abstractmethod
    def _update_model():
        """Update model params"""
        raise NotImplementedError("")

    @abstractmethod
    def _report_results():  # set_scores
        """Report result to HPO"""
        raise NotImplementedError("")

    @abstractmethod
    def run_algo(self, obj_file, template_path):
        """Interface for the HPO to run the training"""
        raise NotImplementedError("")


class NniWrapper(HpoWrap):
    """
    Wrapper for NNI
    """

    def __init__(self, algo_dict=None, params=None):
        """
        Args:
            output_folder: file paths to copy the algo scripts
            algo: an dict that has {name: Algo object} format.
                The Algo object must have get_scripts_path method
            override:
                a set of parameter to override the HPO config, e.g. search space
        """
        self.algo = None
        self.task_prefix = None


        if algo_dict is not None:
            if len(algo_dict.keys()) > 1:
                raise ValueError(f"object {algo_dict} only allows 1 key, but there are {len(algo.keys())}")
            name = list(algo_dict.keys())[0]  # the only key is the name of the model

            algo = algo_dict[name]

            base_task_dir = algo.get_output_path()
            if params is None:
                obj_bytes = pickle.dumps(algo)
            else:
                base_task_dir += "_hpo_override"
                task_name = os.path.basename(base_task_dir)
                output_folder = os.path.dirname(base_task_dir)
                algo_override = deepcopy(algo)  # avoid overriding the existing algo
                algo_override.export_to_disk(output_folder, task_name, **params)
                obj_bytes = pickle.dumps(algo_override)
            
            obj_file = os.path.join(base_task_dir, 'algo_object.pkl')
            with open(obj_file, "wb") as f_pi:
                f_pi.write(obj_bytes)

            logger.info("Add the following line in the trialCommand in your NNI config: ")
            logger.info(f"python -m monai.apps.auto3dseg NniWrapper run_algo {base_task_dir} folder/to/hpo/results/")

    def _get_hyperparameters(self):
        """
        Get parameter for next round of training from nni server
        """
        return nni.get_next_parameter()

    def _update_params(self, params):  #generate
        """
        Translate the parameter from monai bundle to nni format
        """
        self.params = params

    def _get_task_id(self):
        task_id = ""
        for k, v in self.params.items():
            task_id += f"_{k}_{v}"
        if len(task_id) == 0:
            task_id = "_None"  # avoid rewriting the original
        return task_id
    
    def generate(self, output_folder='.'):
        task_id = self._get_task_id()
        if hasattr(self.algo, 'export_to_disk') and callable(getattr(self.algo, 'export_to_disk')):
            self.algo.export_to_disk(output_folder, self.task_prefix + task_id)
        else:
            ConfigParser.export_config_file(self.params, os.path.join(output_folder, self.task_prefix + task_id))

    def _report_results(self, acc):  # set_score
        """
        Report the acc to nni server
        """
        nni.report_final_result(acc)
        return


    def run_algo(self, base_task_dir, output_folder="."):
        """
        The python interface for NNI to run

        Args:
            base_task_dir: 
            output_folder: the root path of the algorithms templates.

        ..code-block:: python
            python -m monai.apps.auto3dseg NniWrapper run_algo "algo.pkl" "template_dir"  #in nni

            on NGC: nnictl create --config hpo_config1.yaml
        """
        # step1 sample hyperparams
        params = self._get_hyperparameters()
        # step 2 update model
        self._update_params(params)
        # step 3 load the algo and train

        if not os.path.isdir(base_task_dir):
            raise ValueError(f"{base_task_dir} is not a directory")
        
        self.task_prefix = os.path.basename(base_task_dir)
        obj_file = os.path.join(base_task_dir, 'algo_object.pkl')
        if not os.path.isfile(obj_file):
            raise ValueError(f"{obj_file} is not found in {base_task_dir}")

        sys.path.insert(0, base_task_dir)

        with open(obj_file, "rb") as f_pi:
            algo_bytes = f_pi.read()
        self.algo = pickle.loads(algo_bytes)

        self.generate(output_folder)

        self.algo.train(self.params)
        acc = self.algo.get_score()
        # step 4 report validation acc to controller
        self._report_results(acc)
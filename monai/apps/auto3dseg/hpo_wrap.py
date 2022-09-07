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

import nni

from monai.auto3dseg.algo_gen import AlgoGen
from monai.bundle.config_parser import ConfigParser

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
    def _report_results():
        """Report result to HPO"""
        raise NotImplementedError("")

    @abstractmethod
    def run_algo(self, obj_file, template_path):
        """Interface for the HPO to run the training"""
        raise NotImplementedError("")


default_nni_config = {
    "experimentName": "dummy",
    "searchSpace": {"lr": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]}},
    "trialCommand": "",  # will be replaced in the wrapper
    "trialCommand": "python -m monai.apps.auto3dseg NniWrapper run_algo ",
    "trialCodeDirectory": "",  # will be replaced in the wrapper
    "trialGpuNumber": 1,
    "trialConcurrency": 1,
    "maxTrialNumber": 10,
    "maxExperimentDuration": "1h",
    "tuner": {"name": "GridSearch"},
    "trainingService": {"platform": "local", "useActiveGpu": True},
}


class NniWrapper(HpoWrap):
    """
    Wrapper for NNI
    """

    def __init__(self, algo=None, **override):
        """
        Args:
            algo: an dict that has {name: Algo object} format.
                The Algo object must have get_script_path method
            override:
                a set of parameter to override the HPO config, e.g. search space
        """
        self.algo = algo
        self.cfg = deepcopy(default_nni_config)
        for k, v in override.items():
            self.cfg.update({k: v})

        if self.algo is not None:
            if len(algo.keys()) > 1:
                raise ValueError(f"object {algo} only allows 1 key, but there are {len(algo.keys())}")
            name = list(algo.keys())[0]  # the only key
            self.cfg.update({"experimentName": name})

            # template path, fixed dir structure
            template_path = os.path.abspath(os.path.join(algo[name].get_script_path(), "..", ".."))
            obj_file = os.path.join(algo[name].get_script_path(), name + ".pkl")

            obj_bytes = pickle.dumps(algo[name])
            with open(obj_file, "wb") as f_pi:
                f_pi.write(obj_bytes)

            self.cfg["trialCommand"] += obj_file + " " + template_path

    def _get_hyperparameters(self):
        """
        Get parameter for next round of training from nni server
        """
        return nni.get_next_parameter()

    def _update_model(self, params):
        """
        Translate the parameter from monai bundle to nni format
        """
        self.params = self.translate_nni_to_bundle(params)

    def _report_results(self, acc):
        """
        Report the acc to nni server
        """
        nni.report_final_result(acc)
        return

    def translate_nni_to_bundle(self, params):
        """Translate the config items from NNI to bundle"""
        return params  # tbd

    def run_algo(self, obj_file, template_path):
        """
        The python interface for NNI to run

        Args:
            obj_file: the pickle dump of the algo object.
            template_path: the root path of the algorithms templates.

        ..code-block:: python
            python -m monai.apps.auto3dseg NniWrapper run_algo "algo.pkl" "template_dir"

        """
        sys.path.insert(0, template_path)
        # step1 sample hyperparams
        params = self._get_hyperparameters()
        # step 2 update model
        self._update_model(params)
        # step 3 train
        with open(obj_file, "rb") as f_pi:
            algo_bytes = f_pi.read()
        algo = pickle.loads(algo_bytes)
        acc = algo.train(self.params)
        print(acc)
        # step 4 report validation acc to controller
        self._report_results(acc)

    def generate(self, output_yaml="dummy_config.yaml"):
        """Write configs for NNI to run"""
        ConfigParser.export_config_file(self.cfg, output_yaml, fmt="yaml")

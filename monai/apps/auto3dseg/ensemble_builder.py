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


import importlib
import logging
import os
import sys
import json

import numpy as np

from abc import abstractmethod, ABC
from typing import Optional, Sequence, Union
from copy import deepcopy

from monai.auto3dseg import concat_val_to_np
from monai.bundle import ConfigParser
from monai.utils.enums import StrEnum


logger = logging.getLogger(__name__)

class ScriptEnsembleKeys(StrEnum):
    """
    Default keys for Mixed Ensemble
    """
    ID = "identifier"
    ALGO = "infer_algo"
    SCORE = "best_metric"


class ScriptEnsemble(ABC):
    def __init__(self):
        self.algos = []
        self.infer_files = []

    def set_algos(self, infer_algos):
        """
        register model in the ensemble
        """
        self.algos = deepcopy(infer_algos)


    def get_algo(self, identifier):
        """ get a model by identifier"""
        for algo in self.algos:
            if identifier == algo[ScriptEnsembleKeys.ID]:
                return algo

    def set_infer_files(self, dataroot, datalist_filename):
        with open(datalist_filename) as f:
            datalist = json.load(f)

        for d in datalist["testing"]:
            self.infer_files.append({"image": os.path.join(dataroot, d["image"])})

    @abstractmethod
    def rank_algos(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, print_results: bool=True, save_predictions: bool=True):
        """
        predict results after the models are ranked/weighted
        """
        raise NotImplementedError


class EnsembleBestN(ScriptEnsemble):
    def __init__(self, n_best: int = 5):
        super().__init__()
        self.n_best = n_best

    def sort_score(self):
        """sort the best_metrics"""
        scores = concat_val_to_np(self.algos, [ScriptEnsembleKeys.SCORE])
        return np.argsort(scores).tolist()


    def rank_algos(self):
        ranks = self.sort_score()
        if len(ranks) < self.n_best:
            raise ValueError("Number of available algos is less than user-defined N")

        # get the indices that the rank is larger than N
        indices = [i for (i, r) in enumerate(ranks) if r >= self.n_best]

        # remove the found indices
        indices = sorted(indices, reverse=True)

        self.ranked_algo = deepcopy(self.algos)
        for idx in indices:
            if idx < len(self.ranked_algo):
                self.ranked_algo.pop(idx)


    def predict(self, **override):

        for i in range(len(self.infer_files)):
            preds = []
            infer_filename = self.infer_files[i]
            for algo in self.ranked_algo:
                infer_instance = algo[ScriptEnsembleKeys.ALGO]
                preds.append(infer_instance.infer(infer_filename))
            output = sum(preds)/len(preds)
            print('ensemble_algorithm, preds', len(preds), 'output shape', output.shape)


class ScriptEnsembleBuilder:
    """
    Build ensemble workflow from configs and arguments.
    Example usage

    builder = ScriptEnsembleBuilder(algo_paths, best_metrics, test_list)
    builder.set_ensemble_method(EnsembleBestN(3))
    ensemble = builder.get_ensemble()

    result = ensemble.predict()

    """

    def __init__(self,
        algo_paths: Sequence[str],
        best_metrics: Sequence[Union[float, None]],
        data_src_cfg_filename: Optional[str] = None,
    ):
        self.infer_algos = []
        self.best_metrics = best_metrics
        self.ensemble: ScriptEnsemble = None
        self.data_src_cfg = ConfigParser(globals=False)

        if len(algo_paths) != len(best_metrics):
            raise ValueError("Numbers of elements in algo_paths and best_metrics are not equal")

        if data_src_cfg_filename is not None and os.path.exists(str(data_src_cfg_filename)):
            self.data_src_cfg.read_config(data_src_cfg_filename)

        for algo_path, best_metric in zip(algo_paths, best_metrics):
            # load inference_config_paths
            # raise warning/error if not found
            if not os.path.isdir(algo_path):
                raise ValueError(f"{algo_path} is not a directory. Please check the path.")

            identifier = os.path.basename(algo_path)
            infer_path = os.path.join(algo_path, 'scripts', 'infer.py')

            if not os.path.isfile(infer_path):
                raise ValueError(f"{infer_path} is not found. Please check the path.")

            config_path = os.path.join(algo_path, 'configs', 'algo_config.yaml')

            config_path = config_path if os.path.isfile(config_path) else None

            self.add_inferer(identifier, infer_path, config_path, best_metric)


    def add_inferer(self,
            identifier: str,
            infer_path: str,
            config_path: str=None,
            best_metric: Optional[float]=None,
            **override):
        """Add model inferer to the builder."""

        # module = importlib.import_module(infer_path)
        # class_ = getattr(module, 'InferClass')
        # algo = class_(config_path, **override)

        spec = importlib.util.spec_from_file_location("InferClass", infer_path)
        infer_class = importlib.util.module_from_spec(spec)
        sys.modules["InferClass"] = infer_class
        spec.loader.exec_module(infer_class)
        infer_instance = infer_class.InferClass(config_path, **override)

        if best_metric is None:
            raise ValueError("Feature to re-valiate is to be implemented")

        algo = {
            ScriptEnsembleKeys.ID: identifier,
            ScriptEnsembleKeys.ALGO: infer_instance,
            ScriptEnsembleKeys.SCORE: best_metric
        }
        self.infer_algos.append(algo)


    def set_ensemble_method(self, ensemble: ScriptEnsemble):
        """Set the ensemble method"""

        ensemble.set_algos(self.infer_algos)
        ensemble.rank_algos()
        ensemble.set_infer_files(self.data_src_cfg["dataroot"], self.data_src_cfg["datalist"])

        self.ensemble = ensemble


    def get_ensemble(self):
        """Get the ensemble"""


        return self.ensemble

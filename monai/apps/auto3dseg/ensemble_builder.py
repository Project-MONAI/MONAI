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


import json
import logging
import os

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Optional, Sequence

import numpy as np

from monai.apps.auto3dseg.bundle_gen import BundleAlgo
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
    """
    The base class of Ensemble methods
    """

    def __init__(self):
        self.algos = []
        self.infer_files = []

    def set_algos(self, infer_algos):
        """
        Register model in the ensemble
        """
        self.algos = deepcopy(infer_algos)

    def get_algo(self, identifier):
        """
        Get a model by identifier.

        Args:
            identifier: the name of the bundleAlgo
        """
        for algo in self.algos:
            if identifier == algo[ScriptEnsembleKeys.ID]:
                return algo

    def set_infer_files(self, dataroot: str, data_src_cfg_file: str):
        """
        Set the files to perform model inference.

        Args:
            dataroot: the path of the files
            data_src_cfg_file: the data source file path
        """
        with open(data_src_cfg_file) as f:
            datalist = json.load(f)

        for d in datalist["testing"]:
            self.infer_files.append({"image": os.path.join(dataroot, d["image"])})

    @abstractmethod
    def rank_algos(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        """
        predict results after the models are ranked/weighted
        """
        raise NotImplementedError


class EnsembleBestN(ScriptEnsemble):
    """
    Ensemble method that select N model out of all using the models' best_metric scores

    Args:
        n_best: number of models to pick for ensemble (N).
    """

    def __init__(self, n_best: int = 5):

        super().__init__()
        self.n_best = n_best

    def sort_score(self):
        """sort the best_metrics"""
        scores = concat_val_to_np(self.algos, [ScriptEnsembleKeys.SCORE])
        return np.argsort(scores).tolist()

    def rank_algos(self):
        """rank the model"""
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

    def predict(self, param: dict = None):
        """predict the ensembled pred"""
        if param is None:
            param = {}

        for i in range(len(self.infer_files)):
            preds = []
            infer_filename = self.infer_files[i]
            for algo in self.ranked_algo:
                infer_instance = algo[ScriptEnsembleKeys.ALGO]
                param.update({"files": [infer_filename]})
                pred = infer_instance.predict(param)
                preds.append(pred[0])
            output = sum(preds) / len(preds)
            print("ensemble_algorithm, preds", len(preds), "output shape", output.shape)


class ScriptEnsembleBuilder:
    """
    Build ensemble workflow from configs and arguments.

    Args:
        history: a collection of trained bundleAlgo algorithms.
        data_src_cfg_filename: filename of the data source.

    Examples:

        ..code-block:: python
            builder = ScriptEnsembleBuilder(history, data_src_cfg)
            builder.set_ensemble_method(EnsembleBestN(3))
            ensemble = builder.get_ensemble()

            result = ensemble.predict()
    """

    def __init__(self, history: Sequence[Dict], data_src_cfg_filename: Optional[str] = None):
        self.infer_algos = []
        self.ensemble: ScriptEnsemble = None
        self.data_src_cfg = ConfigParser(globals=False)

        if data_src_cfg_filename is not None and os.path.exists(str(data_src_cfg_filename)):
            self.data_src_cfg.read_config(data_src_cfg_filename)

        for h in history:
            # load inference_config_paths
            # raise warning/error if not found
            if len(h.keys()) > 1:
                raise ValueError(f"{h} should only contain one set of genAlgo key-value")

            name = list(h.keys())[0]
            gen_algo = h[name]
            best_metric = gen_algo.get_score()
            algo_path = gen_algo.output_path
            infer_path = os.path.join(algo_path, "scripts", "infer.py")

            if not os.path.isdir(algo_path):
                raise ValueError(f"{gen_algo.output_path} is not a directory. Please check the path.")

            if not os.path.isfile(infer_path):
                raise ValueError(f"{infer_path} is not found. Please check the path.")

            self.add_inferer(name, gen_algo, best_metric)

    def add_inferer(self, identifier: str, gen_algo: BundleAlgo, best_metric: Optional[float] = None):
        """
        Add model inferer to the builder.

        Args:
            identifier: name of the bundleAlgo.
            gen_algo: a trained BundleAlgo model object.
            best_metric: the best metric in validation of the trained model.
        """

        if best_metric is None:
            raise ValueError("Feature to re-valiate is to be implemented")

        algo = {
            ScriptEnsembleKeys.ID: identifier,
            ScriptEnsembleKeys.ALGO: gen_algo,
            ScriptEnsembleKeys.SCORE: best_metric,
        }
        self.infer_algos.append(algo)

    def set_ensemble_method(self, ensemble: ScriptEnsemble):
        """
        Set the ensemble method.

        Args:
            ensemble: the ScriptEnsemble to build.
        """

        ensemble.set_algos(self.infer_algos)
        ensemble.rank_algos()
        ensemble.set_infer_files(self.data_src_cfg["dataroot"], self.data_src_cfg["datalist"])

        self.ensemble = ensemble

    def get_ensemble(self):
        """Get the ensemble"""

        return self.ensemble

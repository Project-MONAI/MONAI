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

import logging
import os

import importlib
from monai.auto3dseg.ensemble_methods import EnsembleMethods, EnsembleBestN
import torch
from monai.bundle import ConfigParser
from typing import Dict, Optional, Union
from monai.losses import DiceCELoss

logger = logging.getLogger(__name__)

class Ensemble:
    """Build ensemble workflow from configs and arguments."""

    def __init__(self,
        algo_folder_paths,
        device: Union[str, torch.device],
        best_metrics: Optional[Dict[str, float]] = None,
        emsemble_weights=None,
        ensemble_nbest=5,
        metric=DiceCELoss
    ):

        inference_default_configs = {}
        self.inferers = []
        self.best_metrics = best_metrics
        self.bundle_inference_config = {}
        self.validate_tasks = []

        for f in algo_folder_paths:
            # load inference_config_paths
            # raise warning/error if not found
            self.add_inferer(os.path.join(f, "Inferer.py"))

            parser = ConfigParser(os.path.join(f, "validate.yaml"))
            model_name = parser['name']
            if model_name not in best_metrics:
                self.validate_tasks.append((model_name, self.inferers[-1]))
            self.bundle_inference_config.update[{model_name, os.path.join(f, "infer.yaml")}]

        self.select_ensemble_method(EnsembleBestN())


    def add_inferer(self, path_to_inference_wrapper, **kwargs):
        """Add model inferer to the builder."""
        module = importlib.import_module(path_to_inference_wrapper)
        class_ = getattr(module, 'InferClass')
        algo = class_(**kwargs)
        self.inferrers.append(algo)


    def select_ensemble_method(self, ensemble: EnsembleMethods):
        """Select the ensemble method"""

        self.ensemble = ensemble


    def __call__(self, **kwargs):
        """Get the score if it hasn't yet, and ensemble."""
        for (name, model) in self.validate_tasks:
            self.best_metrics.update({name, model(self.datsets['validate'])})

        for i, _network in enumerate(self.inferers):
            _network.load_state_dict(torch.load(self.bundle_path+f"/models/model{i}.pt"))
            self.ensemble.get_model(_network)

        self.ensemble.predict(**kwargs)

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
import argparse
import subprocess

import importlib
from monai import inferers
import monai.bundle
from monai.auto3dseg.ensemble import Ensemble, EnsembleBestN
import torch
from monai.bundle import ConfigParser
from monai.engines import EnsembleEvaluator
from monai.utils import optional_import
from typing import List, Dict, Any, Optional, Union
from monai.losses import DiceCELoss
logger = logging.getLogger(__name__)

class EnsembleBuilder:
    """
    Examples:

        ..code-block::python
            
            # init
            Ensemble(algo_folder_paths: List[str])


    """
    def __init__(self, 
        algo_folder_paths, 
        algo_metric_results: List[Dict[str, Any]], 
        device: Union[str, torch.device], 
        ensemble_nbest=5, 
        performances: Optional[Dict[str, float]] = None,
        metric=DiceCELoss
    ): 
        
        inference_default_configs = {}
        self.inferers = []
        self.performances = performances
        self.bundle_inference_config = {}
        self.validate_tasks = []

        for f in algo_folder_paths:
            # load inference_config_paths
            # raise warning/error if not found
            self.add_inferer(os.path.join(f, "Inferer.py"))
            
            parser = ConfigParser(os.path.join(f, "validate.yaml"))
            model_name = parser['name']
            if model_name not in performances:
                self.validate_tasks.append((model_name, self.inferers[-1]))
            self.bundle_inference_config.update[{model_name, os.path.join(f, "infer.yaml")}]

    
    def add_inferer(self, path_to_inference_wrapper, **kwargs):
        """
        """
        module = importlib.import_module(path_to_inference_wrapper)
        class_ = getattr(module, 'InferClass')
        algo = class_(**kwargs)
        self.inferrers.append(algo)


    def run(self):
        for (name, model) in self.validate_tasks:
            self.performances.update({name, model(self.datsets['validate'])})
        
        networks = []
        for i, _network in enumerate(self.inferers):
            _network.load_state_dict(torch.load(self.bundle_path+f"/models/model{i}.pt"))
            networks.append(_network)

        evaluator = EnsembleEvaluator(
            device=self.device,
            val_data_loader=self.bundle_inference_config.get_parsed_content("dataloader"),
            pred_keys=["pred", "pred", "pred", "pred", "pred"],
            networks=networks,
            inferer=self.bundle_inference_config.get_parsed_content("inferer"),
            postprocessing=self.bundle_inference_config.get_parsed_content("postprocessing"),
        )
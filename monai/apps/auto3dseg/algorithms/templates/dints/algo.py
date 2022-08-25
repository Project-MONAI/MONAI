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
import os
import sys

from monai.apps.auto3dseg.algorithms.algorithm import Algorithm


class Algo(Algorithm):
    def __init__(self, algorithm_dir: str = None):
        Algorithm.__init__(self, algorithm_dir)

        self.best_metric = -1.0
        self.params = {}

        config_file = [
            "configs/hyper_parameters.yaml",
            "configs/network.yaml",
            "configs/transforms_train.yaml",
            "configs/transforms_validate.yaml",
        ]
        self.config_file = [
            os.path.join(self.algorithm_dir, _item) for _item in config_file
        ]

    def train(self, module_path: str = ""):
        if module_path == "":
            module_path = "{0:s}.scripts.train".format(
                self.algorithm_dir.split(os.sep)[-1]
            )

        module = importlib.import_module(module_path)
        func_ = getattr(module, "run")

        if len(self.params) == 0:
            self.best_metric = func_(config_file=self.config_file)
        else:
            self.best_metric = func_(config_file=self.config_file, **self.params)

        return self.best_metric

    def update(self, params: dict = {}):
        self.params = params

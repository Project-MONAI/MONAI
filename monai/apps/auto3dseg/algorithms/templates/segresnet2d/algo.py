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

import glob
import importlib
import os
import subprocess
import torch
import yaml

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
        if torch.cuda.device_count() > 1:
            cmd = "torchrun --nnodes={0:d} --nproc_per_node={1:d} -m {2:s}.scripts.train run --config_file ['{2:s}/configs/hyper_parameters.yaml','{2:s}/configs/network.yaml','{2:s}/configs/transforms_train.yaml','{2:s}/configs/transforms_validate.yaml']".format(
                1,
                torch.cuda.device_count(),
                self.algorithm_dir.split(os.sep)[-1],
            )

            if len(self.params) > 0:
                for _k in self.params:
                    cmd += " --" + str(_k) + " " + str(self.params[_k])

            _ = subprocess.run(cmd.split(), check=True)

            list_of_files = glob.glob(os.path.join(self.algorithm_dir.split(os.sep)[-1], "*", "progress.yaml"))
            latest_file = max(list_of_files, key=os.path.getctime)

            with open(latest_file, "r") as stream:
                progress_data = yaml.safe_load(stream)
            self.best_metric = progress_data[-1]["best_avg_dice_score"]

        else:
            if module_path == "":
                module_path = "{:s}.scripts.train".format(
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

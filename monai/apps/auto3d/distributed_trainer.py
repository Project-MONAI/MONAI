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

"""
Step 3 of the AutoML pipeline.
"""


import subprocess
import sys
from os import path


class DistributedTrainer:
    def __init__(self, configer, n_fold: int = 5):
        if sys.platform == "win32":
            raise ValueError("unsupported platform")
        else:
            self.train_sh = path.join(configer["script_path"], "train.sh")
            self.folder = configer["script_path"]
            self.n_gpu = 8 if configer["multigpu"] else 1
            self.n_fold = n_fold

    def train(self):
        for fold in range(self.n_fold):
            cmd = f"bash {self.train_sh} {fold} {self.folder} {self.n_gpu}"
            subprocess.run(cmd.split(), check=True)

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

from abc import abstractmethod
import importlib
import os
class HPO_wrapper():
    def __init__(self, algo_name, task_folder, task_module):
        self.algo = self._get_trainer(algo_name, task_folder)
        self.algo_name = algo_name
        self.task_folder = task_folder
        self.task_module = task_module

    def _get_trainer(self, algo_name, task_folder):
        module = importlib.import_module(
            f"monai.apps.auto3dseg.algorithms.templates.{algo_name:s}.algo"
        )
        class_ = getattr(module, "Algo")
        algo = class_(algorithm_dir=os.path.join(task_folder, algo_name))
        return algo

    @abstractmethod
    def _get_hyperparameters():
        pass

    @abstractmethod
    def _update_algo():
        pass

    @abstractmethod
    def _report_results():
        pass

    def __call__():
        pass

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

from typing import Optional, Tuple, Union, Dict

from monai.apps.auto3d.algorithm_autoconfig import auto_configer
from monai.apps.auto3d.data_analyzer import DataAnalyzer
from monai.apps.auto3d.distributed_trainer import DistributedTrainer

class AutoRunner:
    """

    """
    def __init__(
        self, 
        dataroot: str,
        datalist: Union[str, Dict],
        input_args: Optional[Tuple] = None,
        ):
        
        self.dataroot = dataroot
        self.datalist = datalist
        self.input_args = self._parse_input_args(**input_args)
    
    def _parse_input_args(
        self, 
        input_args: Tuple = None
        ):
        # name: Optional[str] = None,
        # task: Optional[str] = "segmentation",
        # modality: Optional[str] = "MRI",
        # multigpu: Optional[bool] = False,
        # output_path: Optional[bool] = None,
        pass

    def automate(
        self,
        param: Dict = None
        ):
        
        analyser = DataAnalyzer(self.datalist, self.dataroot)
        datastat = analyser.get_all_case_stats()

        self.input_args['datastat'] = datastat

        networks = ["UNet"]
        configs = []
        for net in networks:
            configer = auto_configer(net, **self.input_args)
            config = configer.generate_scripts()
            configs.append(config)
        
        models_path = []
        for config in configs:
            model = DistributedTrainer(config)
            model.train(self.input_args['output_path'])
            models_path.append()
        
        return models_path

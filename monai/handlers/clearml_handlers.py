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

from __future__ import annotations

from .tensorboard_handlers import TensorBoardHandler, TensorBoardImageHandler, TensorBoardStatsHandler


class ClearMLStatsHandler(TensorBoardStatsHandler):
    """
    Class to write tensorboard stats into ClearML WebUI.
    """

    def __init__(self, project_name ="Muhammad/monai", task_name="unet_training_dict", output_uri=True, *args, **kwargs):
        try:
            from clearml import Task
        except ModuleNotFoundError:
            raise "Please install ClearML with 'pip install clearml' before using it."

        if Task.current_task():
            self.clearml_task = Task.current_task()
        else:
            self.clearml_task = Task.init(project_name=project_name, task_name=task_name, output_uri=output_uri)
        
        super().__init__(*args, **kwargs)


class ClearMLImageHandler(TensorBoardImageHandler):
    """
    Class to write tensorboard image stats into ClearML WebUI.
    """

    def __init__(self, project_name ="Muhammad/monai", task_name="unet_training_dict", output_uri=True, *args, **kwargs):
        try:
            from clearml import Task
        except ModuleNotFoundError:
            raise "Please install ClearML with 'pip install clearml' before using it."

        if Task.current_task():
            self.clearml_task = Task.current_task()
        else:
            self.clearml_task = Task.init(project_name=project_name, task_name=task_name, output_uri=output_uri)
        
        super().__init__(*args, **kwargs)

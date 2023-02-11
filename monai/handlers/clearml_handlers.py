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
from .tensorboard_handlers import TensorBoardImageHandler, TensorBoardStatsHandler


class ClearMLHandler:
    """

    Base class for the handlers to log everything to ClearML WebUI.

    Args:
        project_name:               The name of the project in which the experiment will be created. If the project does not 
                                    exist, it is created. If project_name is None, the repository name is used. (Optional)
        task_name:                  ClearML task name. Default set to 'monai clearml example'.
        output_uri:                 The default location for output models and other artifacts. 
                                    If True, the default files_server will be used for model storage. In the default location, ClearML creates a subfolder for the output. 
                                    The subfolder structure is the following: <output destination name> / <project name> / <task name>.<Task ID>. Default set to 'True'
        tags:                       Add a list of tags (str) to the created Task. For example: tags=['512x512', 'yolov3']. Default set to empty Python List.
        reuse_last_task_id:         Force a new Task (experiment) with a previously used Task ID, and the same project and Task name. Default set to 'True'.
        conitnue_last_task:         Continue the execution of a previously executed Task (experiment). Default set to 'False'.
        auto_connect_frameworks:    Automatically connect frameworks This includes patching MatplotLib, XGBoost, scikit-learn, Keras callbacks, and TensorBoard/X to 
                                    serialize plots, graphs, and the model location to the ClearML Server (backend), in addition to original output destination.

    For more details of ClearML usage, please refer to: https://clear.ml/docs/latest/docs/references/sdk/task
    
    """

    def __init__(self, project_name:str, task_name:str, output_uri:str, tags:List[str], reuse_last_task_id:bool, continue_last_task:bool, auto_connect_frameworks):
        try:
            from clearml import Task
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Please install ClearML with 'pip install clearml' before using it.")

        if Task.current_task():
            self.clearml_task = Task.current_task()
        else:
            self.clearml_task = Task.init(
                                            project_name=project_name, 
                                            task_name=task_name, 
                                            output_uri=output_uri,
                                            tags=tags,
                                            reuse_last_task_id=reuse_last_task_id,
                                            continue_last_task=continue_last_task,
                                            auto_connect_frameworks=auto_connect_frameworks
                                        )


class ClearMLStatsHandler(ClearMLHandler,TensorBoardStatsHandler):
    """

    Class to write tensorboard stats by inheriting TensorBoardStatsHandler class. 
    Everything from Tensorboard is logged automatically to ClearML WebUI.

    """

    def __init__(
                    self, 
                    project_name:str = None, 
                    task_name:str = None,
                    output_uri:str = True,
                    tags:List[str] = [],
                    reuse_last_task_id:bool = True,
                    continue_last_task:bool = False,
                    auto_connect_frameworks = True,
                    *args, 
                    **kwargs
                ):
        
        """ 
        Args:
            project_name:               The name of the project in which the experiment will be created. If the project does not 
                                        exist, it is created. If project_name is None, the repository name is used. (Optional)
            task_name:                  ClearML task name. Default set to 'monai clearml example'.
            output_uri:                 The default location for output models and other artifacts. 
                                        If True, the default files_server will be used for model storage. In the default location, ClearML creates a subfolder for the output. 
                                        The subfolder structure is the following: <output destination name> / <project name> / <task name>.<Task ID>. Default set to 'True'.
            tags:                       Add a list of tags (str) to the created Task. For example: tags=['512x512', 'yolov3']. Default set to empty Python List.
            reuse_last_task_id:         Force a new Task (experiment) with a previously used Task ID, and the same project and Task name. Default set to 'True'.
            conitnue_last_task:         Continue the execution of a previously executed Task (experiment). Default set to 'False'.
            auto_connect_frameworks:    Automatically connect frameworks This includes patching MatplotLib, XGBoost, scikit-learn, Keras callbacks, and TensorBoard/X to 
                                        serialize plots, graphs, and the model location to the ClearML Server (backend), in addition to original output destination.
        """

        ClearMLHandler.__init__(self,
                                project_name=project_name, 
                                task_name=task_name, 
                                output_uri=output_uri,
                                tags=tags,
                                reuse_last_task_id=reuse_last_task_id,
                                continue_last_task=continue_last_task,
                                auto_connect_frameworks=auto_connect_frameworks
                                )
        TensorBoardStatsHandler.__init__(self, *args, **kwargs)


class ClearMLImageHandler(ClearMLHandler,TensorBoardImageHandler):
    """
    
    This class inherits all functionality from TensorBoardImageHandler class. 
    Everything from Tensorboard is logged automatically to ClearML WebUI.
    
    """

    def __init__(
                    self, 
                    project_name:str = None, 
                    task_name:str = None, 
                    output_uri:str = True,
                    tags:List[str] = [],
                    reuse_last_task_id:bool = True,
                    continue_last_task:bool = False,
                    auto_connect_frameworks = True,
                    *args, 
                    **kwargs
                ):
        
        """
        Args:
            project_name:               The name of the project in which the experiment will be created. If the project does not 
                                        exist, it is created. If project_name is None, the repository name is used. (Optional)
            task_name:                  ClearML task name. Default set to 'monai clearml example'.
            output_uri:                 The default location for output models and other artifacts. 
                                        If True, the default files_server will be used for model storage. In the default location, ClearML creates a subfolder for the output. 
                                        The subfolder structure is the following: <output destination name> / <project name> / <task name>.<Task ID>. Default set to 'True'.
            tags:                       Add a list of tags (str) to the created Task. For example: tags=['512x512', 'yolov3']. Default set to empty Python List.
            reuse_last_task_id:         Force a new Task (experiment) with a previously used Task ID, and the same project and Task name. Default set to 'True'.
            conitnue_last_task:         Continue the execution of a previously executed Task (experiment). Default set to 'False'.
            auto_connect_frameworks:    Automatically connect frameworks This includes patching MatplotLib, XGBoost, scikit-learn, Keras callbacks, and TensorBoard/X to 
                                        serialize plots, graphs, and the model location to the ClearML Server (backend), in addition to original output destination.
        """

        ClearMLHandler.__init__(self,
                                project_name=project_name, 
                                task_name=task_name, 
                                output_uri=output_uri,
                                tags=tags,
                                reuse_last_task_id=reuse_last_task_id,
                                continue_last_task=continue_last_task,
                                auto_connect_frameworks=auto_connect_frameworks
                                )
        
        TensorBoardImageHandler.__init__(self, *args, **kwargs)

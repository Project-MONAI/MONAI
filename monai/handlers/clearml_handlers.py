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

from typing import TYPE_CHECKING, Mapping, Sequence, Union

from monai.utils import optional_import

from .tensorboard_handlers import TensorBoardImageHandler, TensorBoardStatsHandler

if TYPE_CHECKING:
    from clearml import Task
else:
    Task, _ = optional_import("clearml", name="Task")


class ClearMLHandler:
    """
    Base class for the handlers to log everything to ClearML WebUI.
    For more details of ClearML usage, please refer to:
    https://clear.ml/docs/latest/docs/references/sdk/task

    """

    def __init__(
        self,
        project_name: str | None,
        task_name: str | None,
        output_uri: str | bool,
        tags: Sequence[str] | None,
        reuse_last_task_id: bool,
        continue_last_task: bool,
        auto_connect_frameworks: Union[bool, Mapping[str, Union[bool, str, list]]],
    ):
        """
        Args:
            project_name:               ClearML project name.
            task_name:                  ClearML task name.
            output_uri:                 The default location for output models and other artifacts.
            tags:                       A list of tags (str) to the created Task.
            reuse_last_task_id:         Force a new Task (experiment) with a previously used Task ID.
            continue_last_task:         Continue the execution of a previously executed Task (experiment).
            auto_connect_frameworks:    Automatically connect frameworks.
        """

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
                auto_connect_frameworks=auto_connect_frameworks,
            )


class ClearMLStatsHandler(ClearMLHandler, TensorBoardStatsHandler):
    """

    Class to write tensorboard stats by inheriting TensorBoardStatsHandler class.
    Everything from Tensorboard is logged automatically to ClearML WebUI.

    """

    def __init__(
        self,
        project_name: str | None = "MONAI",
        task_name: str | None = "Default Task",
        output_uri: str | bool = True,
        tags: Sequence[str] | None = None,
        reuse_last_task_id: bool = True,
        continue_last_task: bool = False,
        auto_connect_frameworks: Union[bool, Mapping[str, Union[bool, str, list]]] = True,
        *args,
        **kwargs,
    ):
        """
        Args:
            project_name:               ClearML project name.
            task_name:                  ClearML task name.
            output_uri:                 The default location for output models and other artifacts.
            tags:                       A list of tags (str) to the created Task.
            reuse_last_task_id:         Force a new Task (experiment) with a previously used Task ID.
            continue_last_task:         Continue the execution of a previously executed Task (experiment).
            auto_connect_frameworks:    Automatically connect frameworks.
        """

        ClearMLHandler.__init__(
            self,
            project_name=project_name,
            task_name=task_name,
            output_uri=output_uri,
            tags=tags,
            reuse_last_task_id=reuse_last_task_id,
            continue_last_task=continue_last_task,
            auto_connect_frameworks=auto_connect_frameworks,
        )
        TensorBoardStatsHandler.__init__(self, *args, **kwargs)


class ClearMLImageHandler(ClearMLHandler, TensorBoardImageHandler):
    """

    This class inherits all functionality from TensorBoardImageHandler class.
    Everything from Tensorboard is logged automatically to ClearML WebUI.

    """

    def __init__(
        self,
        project_name: str | None = "MONAI",
        task_name: str | None = "Default Task",
        output_uri: str | bool = True,
        tags: Sequence[str] | None = None,
        reuse_last_task_id: bool = True,
        continue_last_task: bool = False,
        auto_connect_frameworks: Union[bool, Mapping[str, Union[bool, str, list]]] = True,
        *args,
        **kwargs,
    ):
        """
        Args:
            project_name:               ClearML project name.
            task_name:                  ClearML task name.
            output_uri:                 The default location for output models and other artifacts.
            tags:                       A list of tags (str) to the created Task.
            reuse_last_task_id:         Force a new Task (experiment) with a previously used Task ID.
            continue_last_task:         Continue the execution of a previously executed Task (experiment).
            auto_connect_frameworks:    Automatically connect frameworks.

        """

        ClearMLHandler.__init__(
            self,
            project_name=project_name,
            task_name=task_name,
            output_uri=output_uri,
            tags=tags,
            reuse_last_task_id=reuse_last_task_id,
            continue_last_task=continue_last_task,
            auto_connect_frameworks=auto_connect_frameworks,
        )

        TensorBoardImageHandler.__init__(self, *args, **kwargs)

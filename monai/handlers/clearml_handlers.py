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

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from monai.utils import optional_import

from .tensorboard_handlers import TensorBoardImageHandler, TensorBoardStatsHandler


class ClearMLHandler:
    """
    Base class for the handlers to log everything to ClearML.
    For more details of ClearML usage, please refer to:
    https://clear.ml/docs/latest/docs/references/sdk/task

    Usage example is available in the tutorial:
    https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unet_segmentation_3d_ignite.ipynb.

    """

    def __init__(
        self,
        project_name: str | None,
        task_name: str | None,
        output_uri: str | bool,
        tags: Sequence[str] | None,
        reuse_last_task_id: bool,
        continue_last_task: bool,
        auto_connect_frameworks: bool | Mapping[str, bool | str | list],
        auto_connect_arg_parser: bool | Mapping[str, bool],
    ) -> None:
        """
        Args:
            project_name: ClearML project name, default to 'MONAI'.
            task_name: ClearML task name, default to 'monai_experiment'.
            output_uri: The default location for output models and other artifacts, default to 'True'.
            tags: Add a list of tags (str) to the created Task, default to 'None'.
            reuse_last_task_id: Force a new Task (experiment) with a previously used Task ID, default to 'True'.
            continue_last_task: Continue the execution of a previously executed Task (experiment), default to 'False'.
            auto_connect_frameworks: Automatically connect frameworks, default to 'True'.
            auto_connect_arg_parser: Automatically connect an argparse object to the Task, default to 'True'.

        """

        if TYPE_CHECKING:
            import clearml
        else:
            clearml, _ = optional_import("clearml")

        # Always check if the user didn't already add a `task.init`` in before
        # if so, use that task, otherwise create a new one.
        if clearml.Task.current_task():
            self.clearml_task = clearml.Task.current_task()
        else:
            self.clearml_task = clearml.Task.init(
                project_name=project_name,
                task_name=task_name,
                output_uri=output_uri,
                tags=tags,
                reuse_last_task_id=reuse_last_task_id,
                continue_last_task=continue_last_task,
                auto_connect_frameworks=auto_connect_frameworks,
                auto_connect_arg_parser=auto_connect_arg_parser,
            )


class ClearMLStatsHandler(ClearMLHandler, TensorBoardStatsHandler):
    """

    Class to write tensorboard stats by inheriting TensorBoardStatsHandler class.
    Everything from Tensorboard is logged automatically to ClearML.

    Usage example is available in the tutorial:
    https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unet_segmentation_3d_ignite.ipynb.

    """

    def __init__(
        self,
        project_name: str | None = "MONAI",
        task_name: str | None = "monai_experiment",
        output_uri: str | bool = True,
        tags: Sequence[str] | None = None,
        reuse_last_task_id: bool = True,
        continue_last_task: bool = False,
        auto_connect_frameworks: bool | Mapping[str, bool | str | list] = True,
        auto_connect_arg_parser: bool | Mapping[str, bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            project_name: ClearML project name, default to 'MONAI'.
            task_name: ClearML task name, default to 'monai_experiment'.
            output_uri: The default location for output models and other artifacts, default to 'True'.
            tags: Add a list of tags (str) to the created Task, default to 'None'.
            reuse_last_task_id: Force a new Task (experiment) with a previously used Task ID, default to 'True'.
            continue_last_task: Continue the execution of a previously executed Task (experiment), default to 'False'.
            auto_connect_frameworks: Automatically connect frameworks, default to 'True'.
            auto_connect_arg_parser: Automatically connect an argparse object to the Task, default to 'True'.

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
            auto_connect_arg_parser=auto_connect_arg_parser,
        )
        TensorBoardStatsHandler.__init__(self, *args, **kwargs)


class ClearMLImageHandler(ClearMLHandler, TensorBoardImageHandler):
    """

    This class inherits all functionality from TensorBoardImageHandler class.
    Everything from Tensorboard is logged automatically to ClearML.

    Usage example is available in the tutorial:
    https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unet_segmentation_3d_ignite.ipynb.

    """

    def __init__(
        self,
        project_name: str | None = "MONAI",
        task_name: str | None = "monai_experiment",
        output_uri: str | bool = True,
        tags: Sequence[str] | None = None,
        reuse_last_task_id: bool = True,
        continue_last_task: bool = False,
        auto_connect_frameworks: bool | Mapping[str, bool | str | list] = True,
        auto_connect_arg_parser: bool | Mapping[str, bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            project_name: ClearML project name, default to 'MONAI'.
            task_name: ClearML task name, default to 'monai_experiment'.
            output_uri: The default location for output models and other artifacts, default to 'True'.
            tags: Add a list of tags (str) to the created Task, default to 'None'.
            reuse_last_task_id: Force a new Task (experiment) with a previously used Task ID, default to 'True'.
            continue_last_task: Continue the execution of a previously executed Task (experiment), default to 'False'.
            auto_connect_frameworks: Automatically connect frameworks, default to 'True'.
            auto_connect_arg_parser: Automatically connect an argparse object to the Task, default to 'True'.

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
            auto_connect_arg_parser=auto_connect_arg_parser,
        )

        TensorBoardImageHandler.__init__(self, *args, **kwargs)

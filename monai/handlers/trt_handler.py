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

from typing import TYPE_CHECKING

from monai.config import IgniteInfo
from monai.networks import trt_wrap
from monai.utils import min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


class TrtHandler:
    """
    TrtHandler acts as an Ignite handler to apply TRT acceleration to the model.
    Usage example::
        handler = TrtHandler(model=model, path="/test/checkpoint.pt", args={"precision": "fp16"})
        handler(trainer)

    Args:
        path: the file path of checkpoint, it should be a PyTorch `pth` file.
        args: dict : unpacked and passed to TrtWrapper().
        submodules : Hierarchical ids of submodules to convert, e.g. 'image_decoder.decoder'
                    If None, TrtWrapper is applied to the whole model and returned.
                    Otherwise, submodules are replaced in-place with TrtWrappers.
    """

    def __init__(
        self,
        model,
        path,
        args=None,
        submodules=None,
        enabled=True
    ):
        self.model = model
        self.path = path
        self.args = args
        self.enabled = enabled
        self.submodules = submodules or [""]

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        self.logger = engine.logger
        engine.add_event_handler(Events.STARTED, self)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.enabled:
            for submodule in self.submodules:
                trt_wrap(self.model, self.path, args=self.args, submodule=submodule)
                self.logger.info(f"Created TRT wrapper for {self.path}.{submodule}")

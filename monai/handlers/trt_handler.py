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

from monai.networks import trt_compile
from monai.utils import IgniteInfo, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


class TrtHandler:
    """
    TrtHandler acts as an Ignite handler to apply TRT acceleration to the model.
    Usage example::
        handler = TrtHandler(model=model, base_path="/test/checkpoint.pt", args={"precision": "fp16"})
        handler.attach(engine)
        engine.run()
    """

    def __init__(self, model, base_path, args=None, submodule=None):
        """
        Args:
            base_path: TRT path basename. TRT plan(s) saved to "base_path[.submodule].plan"
            args: passed to trt_compile(). See trt_compile() for details.
            submodule : Hierarchical ids of submodules to convert, e.g. 'image_decoder.decoder'
        """
        self.model = model
        self.base_path = base_path
        self.args = args
        self.submodule = submodule

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
        trt_compile(self.model, self.base_path, args=self.args, submodule=self.submodule, logger=self.logger)

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

import logging
import os
import threading
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from monai.config import DtypeLike, IgniteInfo
from monai.utils import min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


class ProbMapProducer:
    """
    Event handler triggered on completing every iteration to calculate and save the probability map
    """

    def __init__(
        self,
        output_dir: str = "./",
        output_postfix: str = "",
        dtype: DtypeLike = np.float64,
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            output_dir: output directory to save probability maps.
            output_postfix: a string appended to all output file names.
            dtype: the data type in which the probability map is stored. Default np.float64.
            name: identifier of logging.logger to use, defaulting to `engine.logger`.

        """
        self.logger = logging.getLogger(name)
        self._name = name
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.dtype = dtype
        self.prob_map: Dict[str, np.ndarray] = {}
        self.counter: Dict[str, int] = {}
        self.num_done_images: int = 0
        self.num_images: int = 0
        self.lock = threading.Lock()

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """

        image_data = engine.data_loader.dataset.image_data  # type: ignore
        self.num_images = len(image_data)

        # Initialized probability maps for all the images
        for sample in image_data:
            name = sample["image"]
            self.counter[name] = sample["num_patches"]
            self.prob_map[name] = np.zeros(sample["map_size"], dtype=self.dtype)

        if self._name is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        if not engine.has_event_handler(self.finalize, Events.COMPLETED):
            engine.add_event_handler(Events.COMPLETED, self.finalize)

    def __call__(self, engine: Engine) -> None:
        """
        This method assumes self.batch_transform will extract metadata from the input batch.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if not isinstance(engine.state.batch, dict) or not isinstance(engine.state.output, dict):
            raise ValueError("engine.state.batch and engine.state.output must be dictionaries.")
        names = engine.state.batch["metadata"]["path"]
        locs = engine.state.batch["metadata"]["map_center"]
        preds = engine.state.output["pred"]
        for name, loc, pred in zip(names, locs, preds):
            self.prob_map[name][loc] = pred
            with self.lock:
                self.counter[name] -= 1
                if self.counter[name] == 0:
                    self.save_prob_map(name)

    def save_prob_map(self, name: str) -> None:
        """
        This method save the probability map for an image, when its inference is finished,
        and delete that probability map from memory.

        Args:
            name: the name of image to be saved.
        """
        file_path = os.path.join(self.output_dir, name)
        np.save(file_path + self.output_postfix + ".npy", self.prob_map[name])

        self.num_done_images += 1
        self.logger.info(f"Inference of '{name}' is done [{self.num_done_images}/{self.num_images}]!")
        del self.prob_map[name]
        del self.counter[name]

    def finalize(self, engine: Engine):
        self.logger.info(f"Probability map is created for {self.num_done_images}/{self.num_images} images!")

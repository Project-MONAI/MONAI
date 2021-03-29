import logging
import os
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from monai.utils import exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")


class ProbMapGenerator:
    """
    Event handler triggered on completing every iteration to save the probability map
    """

    def __init__(
        self,
        output_dir: str = "./",
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            output_dir: output directory to save probability maps.
            name: identifier of logging.logger to use, defaulting to `engine.logger`.

        """
        self.logger = logging.getLogger(name)
        self._name = name
        self.output_dir = output_dir
        self.prob_map: Dict[str, np.ndarray] = {}
        self.level: Dict[str, int] = {}
        self.counter: Dict[str, int] = {}
        self.num_done_images: int = 0
        self.num_images: int = 0

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """

        self.num_images = len(engine.data_loader.dataset.data_list)

        for sample in engine.data_loader.dataset.data_list:
            name = sample["name"]
            self.prob_map[name] = np.zeros(sample["mask_shape"])
            self.counter[name] = len(sample["mask_locations"])
            self.level[name] = sample["level"]

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
        names = engine.state.batch["name"]
        locs = engine.state.batch["mask_location"]
        pred = engine.state.output["pred"]
        for i, name in enumerate(names):
            self.prob_map[name][locs[0][i], locs[1][i]] = pred[i]
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
        np.save(file_path + ".npy", self.prob_map[name])

        self.num_done_images += 1
        self.logger.info(f"Inference of '{name}' is done [{self.num_done_images}/{self.num_images}]!")
        del self.prob_map[name]
        del self.counter[name]
        del self.level[name]

    def finalize(self, engine: Engine):
        if self.counter:
            raise RuntimeError(f"Counter: {self.counter}")
        else:
            self.logger.info(f"Probability map is created for {self.num_done_images}/{self.num_images} images!")

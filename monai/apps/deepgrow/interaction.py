# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
from typing import Dict

import torch

from monai.engines.utils import CommonKeys
from monai.engines.workflow import Engine, Events
from monai.transforms import Compose


class Interaction:
    """
    Ignite handler used to introduce interactions (simulation of clicks) for Deepgrow Training/Evaluation.

    Args:
        transforms: execute additional transformation during every iteration (before train).
            Typically, several Tensor based transforms composed by `Compose`.
        max_interactions: maximum number of interactions per iteration
        train: training or evaluation
        key_probability: field name to fill probability for every interaction
    """

    def __init__(self, transforms, max_interactions: int, train: bool, key_probability: str = "probability") -> None:
        self.transforms = transforms
        self.max_interactions = max_interactions
        self.train = train
        self.key_probability = key_probability

        if not isinstance(self.transforms, Compose):
            transforms = []
            for t in self.transforms:
                transforms.append(self.init_external_class(t))
            self.transforms = Compose(transforms)

    @staticmethod
    def init_external_class(config_dict):
        class_args = None if config_dict.get("args") is None else dict(config_dict.get("args"))
        class_path = config_dict.get("path", config_dict["name"])

        module_name, class_name = class_path.rsplit(".", 1)
        m = importlib.import_module(module_name)
        c = getattr(m, class_name)
        return c(**class_args) if class_args else c()

    def attach(self, engine: Engine) -> None:
        if not engine.has_event_handler(self, Events.ITERATION_STARTED):
            engine.add_event_handler(Events.ITERATION_STARTED, self)

    def __call__(self, engine: Engine, batchdata: Dict[str, torch.Tensor]):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        for j in range(self.max_interactions):
            inputs, _ = engine.prepare_batch(batchdata)
            inputs = inputs.to(engine.state.device)

            engine.network.eval()
            with torch.no_grad():
                if engine.amp:
                    with torch.cuda.amp.autocast():
                        predictions = engine.inferer(inputs, engine.network)
                else:
                    predictions = engine.inferer(inputs, engine.network)

            batchdata.update({CommonKeys.PRED: predictions})
            batchdata[self.key_probability] = torch.as_tensor(
                ([1.0 - ((1.0 / self.max_interactions) * j)] if self.train else [1.0]) * len(inputs)
            )
            batchdata = self.transforms(batchdata)

        return engine._iteration(engine, batchdata)

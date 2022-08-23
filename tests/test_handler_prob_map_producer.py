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

import os
import unittest

import numpy as np
import torch
from ignite.engine import Engine
from parameterized import parameterized
from torch.utils.data import DataLoader

from monai.data.dataset import Dataset
from monai.engines import Evaluator
from monai.handlers import ProbMapProducer, ValidationHandler
from monai.utils.enums import ProbMapKeys

TEST_CASE_0 = ["temp_image_inference_output_1", 2]
TEST_CASE_1 = ["temp_image_inference_output_2", 9]
TEST_CASE_2 = ["temp_image_inference_output_3", 100]


class TestDataset(Dataset):
    def __init__(self, name, size):
        super().__init__(
            data=[
                {
                    "image": name,
                    ProbMapKeys.COUNT.value: size,
                    ProbMapKeys.SIZE.value: np.array([size, size]),
                    ProbMapKeys.LOCATION.value: np.array([i, i + 1]),
                }
                for i in range(size - 1)
            ]
        )
        self.image_data = [
            {
                ProbMapKeys.NAME.value: name,
                ProbMapKeys.COUNT.value: size,
                ProbMapKeys.SIZE.value: np.array([size, size]),
            }
        ]

    def __getitem__(self, index):
        return {
            "image": np.zeros((3, 2, 2)),
            ProbMapKeys.COUNT.value: self.data[index][ProbMapKeys.COUNT.value],
            "metadata": {
                ProbMapKeys.NAME.value: self.data[index]["image"],
                ProbMapKeys.SIZE.value: self.data[index][ProbMapKeys.SIZE.value],
                ProbMapKeys.LOCATION.value: self.data[index][ProbMapKeys.LOCATION.value],
            },
            "pred": index + 1,
        }


class TestEvaluator(Evaluator):
    def _iteration(self, engine, batchdata):
        return batchdata


class TestHandlerProbMapGenerator(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2])
    def test_prob_map_generator(self, name, size):
        # set up dataset
        dataset = TestDataset(name, size)
        data_loader = DataLoader(dataset, batch_size=1)

        # set up engine
        def inference(enging, batch):
            pass

        engine = Engine(inference)

        # add ProbMapGenerator() to evaluator
        output_dir = os.path.join(os.path.dirname(__file__), "testing_data")
        prob_map_gen = ProbMapProducer(output_dir=output_dir)

        evaluator = TestEvaluator(torch.device("cpu:0"), data_loader, size, val_handlers=[prob_map_gen])

        # set up validation handler
        validation = ValidationHandler(interval=1, validator=None)
        validation.attach(engine)
        validation.set_validator(validator=evaluator)

        engine.run(data_loader)

        prob_map = np.load(os.path.join(output_dir, name + ".npy"))
        self.assertListEqual(np.vstack(prob_map.nonzero()).T.tolist(), [[i, i + 1] for i in range(size - 1)])
        self.assertListEqual(prob_map[prob_map.nonzero()].tolist(), [i + 1 for i in range(size - 1)])


if __name__ == "__main__":
    unittest.main()

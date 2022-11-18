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

import unittest

import torch

from monai.engines import PrepareBatchDefault, SupervisedEvaluator
from tests.utils import assert_allclose


class TestNet(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x


class TestPrepareBatchDefault(unittest.TestCase):
    def test_dict_content(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = [
            {
                "image": torch.tensor([1, 2]),
                "label": torch.tensor([3, 4]),
                "extra1": torch.tensor([5, 6]),
                "extra2": 16,
                "extra3": "test",
            }
        ]
        # set up engine
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=dataloader,
            epoch_length=1,
            network=TestNet(),
            non_blocking=False,
            prepare_batch=PrepareBatchDefault(),
            decollate=False,
            mode="eval",
        )
        evaluator.run()
        output = evaluator.state.output
        assert_allclose(output["image"], torch.tensor([1, 2], device=device))
        assert_allclose(output["label"], torch.tensor([3, 4], device=device))

    def test_tensor_content(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = [torch.tensor([1, 2])]

        # set up engine
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=dataloader,
            epoch_length=1,
            network=torch.nn.Identity(),
            non_blocking=False,
            prepare_batch=PrepareBatchDefault(),
            decollate=False,
            mode="eval",
        )
        evaluator.run()
        output = evaluator.state.output
        assert_allclose(output["image"], torch.tensor([1, 2], device=device))
        self.assertTrue(output["label"] is None)

    def test_pair_content(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = [(torch.tensor([1, 2]), torch.tensor([3, 4]))]

        # set up engine
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=dataloader,
            epoch_length=1,
            network=torch.nn.Identity(),
            non_blocking=False,
            prepare_batch=PrepareBatchDefault(),
            decollate=False,
            mode="eval",
        )
        evaluator.run()
        output = evaluator.state.output
        assert_allclose(output["image"], torch.tensor([1, 2], device=device))
        assert_allclose(output["label"], torch.tensor([3, 4], device=device))

    def test_empty_data(self):
        dataloader = []
        evaluator = SupervisedEvaluator(
            val_data_loader=dataloader,
            device=torch.device("cpu"),
            epoch_length=0,
            network=TestNet(),
            non_blocking=False,
            prepare_batch=PrepareBatchDefault(),
            decollate=False,
        )
        evaluator.run()


if __name__ == "__main__":
    unittest.main()

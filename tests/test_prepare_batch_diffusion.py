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

import unittest

import torch
from parameterized import parameterized
from monai.engines.utils import DiffusionPrepareBatch
from monai.networks.nets import DiffusionModelUNet
from monai.engines import SupervisedEvaluator
from tests.utils import assert_allclose

TEST_CASES = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [8],
            "norm_num_groups": 8,
            "attention_levels": [True],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        (2, 1, 8, 8),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [8],
            "norm_num_groups": 8,
            "attention_levels": [True],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        (2, 1, 8, 8, 8),
    ],
]



class TestPrepareBatchDiffusion(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_content(self, input_args, image_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = [
            {
                "image": torch.randn(image_size).to(device),
            }
        ]
        # set up engine
        network = DiffusionModelUNet(**input_args).to(device)
        num_train_timesteps = 10
        scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        inferer = DiffusionInferer()
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=dataloader,
            epoch_length=1,
            network=network,
            non_blocking=True,
            prepare_batch=DiffusionPrepareBatch(num_train_timesteps=20),
            decollate=False,
        )
        evaluator.run()
        output = evaluator.state.output
        assert_allclose(output["image"], torch.tensor([1, 2], device=device))
        for k, v in output["pred"].items():
            if isinstance(v, torch.Tensor):
                assert_allclose(v, expected_value[k].to(device))
            else:
                self.assertEqual(v, expected_value[k])


if __name__ == "__main__":
    unittest.main()

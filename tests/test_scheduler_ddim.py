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

from monai.networks.schedulers import DDIMScheduler
from tests.test_utils import assert_allclose

TEST_2D_CASE = []
for beta_schedule in ["linear_beta", "scaled_linear_beta"]:
    TEST_2D_CASE.append([{"schedule": beta_schedule}, (2, 6, 16, 16), (2, 6, 16, 16)])

TEST_3D_CASE = []
for beta_schedule in ["linear_beta", "scaled_linear_beta"]:
    TEST_3D_CASE.append([{"schedule": beta_schedule}, (2, 6, 16, 16, 16), (2, 6, 16, 16, 16)])

TEST_CASES = TEST_2D_CASE + TEST_3D_CASE

TEST_FULl_LOOP = [
    [{"schedule": "linear_beta"}, (1, 1, 2, 2), torch.Tensor([[[[-0.9579, -0.6457], [0.4684, -0.9694]]]])]
]


class TestDDPMScheduler(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_add_noise(self, input_param, input_shape, expected_shape):
        scheduler = DDIMScheduler(**input_param)
        scheduler.set_timesteps(num_inference_steps=100)
        original_sample = torch.zeros(input_shape)
        noise = torch.randn_like(original_sample)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (original_sample.shape[0],)).long()

        noisy = scheduler.add_noise(original_samples=original_sample, noise=noise, timesteps=timesteps)
        self.assertEqual(noisy.shape, expected_shape)

    @parameterized.expand(TEST_CASES)
    def test_step_shape(self, input_param, input_shape, expected_shape):
        scheduler = DDIMScheduler(**input_param)
        scheduler.set_timesteps(num_inference_steps=100)
        model_output = torch.randn(input_shape)
        sample = torch.randn(input_shape)
        output_step = scheduler.step(model_output=model_output, timestep=500, sample=sample)
        self.assertEqual(output_step[0].shape, expected_shape)
        self.assertEqual(output_step[1].shape, expected_shape)

    @parameterized.expand(TEST_FULl_LOOP)
    def test_full_timestep_loop(self, input_param, input_shape, expected_output):
        scheduler = DDIMScheduler(**input_param)
        scheduler.set_timesteps(50)
        torch.manual_seed(42)
        model_output = torch.randn(input_shape)
        sample = torch.randn(input_shape)
        for t in range(50):
            sample, _ = scheduler.step(model_output=model_output, timestep=t, sample=sample)
        assert_allclose(sample, expected_output, rtol=1e-3, atol=1e-3)

    def test_set_timesteps(self):
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=100)
        self.assertEqual(scheduler.num_inference_steps, 100)
        self.assertEqual(len(scheduler.timesteps), 100)

    def test_set_timesteps_with_num_inference_steps_bigger_than_num_train_timesteps(self):
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(num_inference_steps=2000)


if __name__ == "__main__":
    unittest.main()

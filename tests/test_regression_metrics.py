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

import unittest
from functools import partial

import numpy as np
import torch
from parameterized import parameterized

from monai.metrics import MAEMetric, MSEMetric, PSNRMetric, RMSEMetric

# TODO: Add parametrised examples similar to test_comput_meandice.py


class TestRegressionMetrics(unittest.TestCase):
    def test_shape(self):
        metrics = [MSEMetric, MAEMetric, RMSEMetric, partial(PSNRMetric, max_val=1.0)]
        basedim = 10
        for ndim in [2, 3, 4]:
            in_tensor = torch.rand((basedim,) * ndim)
            for mt_fn in metrics:
                out_tensor = mt_fn(reduction="mean")(in_tensor, in_tensor)
                self.assertTrue(len(out_tensor.shape) == 1)

                out_tensor = mt_fn(reduction="sum")(in_tensor, in_tensor)
                self.assertTrue(len(out_tensor.shape) == 0)

                out_tensor = mt_fn(reduction="mean_channel")(in_tensor, in_tensor)
                self.assertTrue(len(out_tensor.shape) == 1 and out_tensor.shape[0] == basedim)

                out_tensor = mt_fn(reduction="sum_channel")(in_tensor, in_tensor)
                self.assertTrue(len(out_tensor.shape) == 1 and out_tensor.shape[0] == basedim)

    def test_ill_shape(self):
        metrics = [MSEMetric, MAEMetric, RMSEMetric, partial(PSNRMetric, max_val=1.0)]
        basedim = 10
        with self.assertRaises(ValueError):
            in_tensor = torch.rand((basedim,))
            for mt_fn in metrics:
                out_tensor = mt_fn()(in_tensor, in_tensor)

    def test_same_input(self):
        metrics = [MSEMetric, MAEMetric, RMSEMetric]
        results = [0.0, 0.0, 0.0]
        basedim = 10
        for ndim in [2, 3, 4]:
            in_tensor = torch.rand((basedim,) * ndim)
            for mt_fn, rs in zip(metrics, results):
                out_tensor = mt_fn(reduction="mean")(in_tensor, in_tensor)
                np.testing.assert_allclose(out_tensor, rs)

        for ndim in [2, 3, 4]:
            in_tensor = torch.rand((basedim,) * ndim)
            out_tensor = PSNRMetric(reduction="mean", max_val=1.0)(in_tensor, in_tensor)
            self.assertTrue(torch.isinf(out_tensor))

    def test_diff_input(self):
        metrics = [MSEMetric, MAEMetric, RMSEMetric]
        results = [1.0, 1.0, 1.0]
        basedim = 10
        for ndim in [2, 3, 4]:
            in_tensor_a = torch.zeros((basedim,) * ndim)
            in_tensor_b = torch.ones((basedim,) * ndim)
            for mt_fn, rs in zip(metrics, results):
                out_tensor = mt_fn(reduction="mean")(in_tensor_a, in_tensor_b)
                np.testing.assert_allclose(out_tensor, rs)

        for ndim in [2, 3, 4]:
            in_tensor_a = torch.zeros((basedim,) * ndim)
            in_tensor_b = torch.ones((basedim,) * ndim)
            out_tensor = PSNRMetric(reduction="mean", max_val=1.0)(in_tensor_a, in_tensor_b)
            np.testing.assert_allclose(out_tensor, 0.0)


if __name__ == "__main__":
    unittest.main()

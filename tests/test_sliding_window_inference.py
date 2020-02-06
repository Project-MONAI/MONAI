# Copyright 2020 MONAI Consortium
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
import numpy as np

from monai.utils.sliding_window_inference import sliding_window_inference


class TestSlidingWindowInference(unittest.TestCase):

    def test_sliding_window_default(self):
        inputs = np.ones((1, 3, 16, 16, 8))
        roi_size = [4, 4, 4]
        sw_batch_size = 4
        device = torch.device("cuda:0")

        def compute(data):
            data = torch.from_numpy(data)
            return data.to(device) + 1

        result = sliding_window_inference(inputs, roi_size, sw_batch_size, compute, device)
        expected_val = torch.ones((1, 3, 16, 16, 8), dtype=torch.float32, device=device) + 1
        self.assertAlmostEqual(result.shape, expected_val.shape)


if __name__ == '__main__':
    unittest.main()

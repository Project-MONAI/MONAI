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
from copy import deepcopy

import numpy as np
from parameterized import parameterized

from monai.transforms import map_classes_to_indices, generate_label_classes_crop_centers, map_and_generate_sampling_centers
from monai.utils.misc import set_determinism
from tests.utils import TEST_NDARRAYS, assert_allclose

class TestMapAndGenerateSamplingCenters(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_map_and_generate_sampling_centers(self, input_data):
        label = np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]])
        spatial_size = (2, 2)
        num_samples = 3
        label_spatial_shape = label.shape
        indices = map_classes_to_indices(**input_data)
        centers = map_and_generate_sampling_centers(label, spatial_size, num_samples, label_spatial_shape, indices)
        self.assertEqual(len(centers), num_samples)

if __name__ == '__main__':
    unittest.main()

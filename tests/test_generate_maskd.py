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

from parameterized import parameterized

from monai.apps.pathology.transforms.post.dictionary import GenerateMaskd
from tests.utils import TEST_NDARRAYS
import numpy as np
import torch

EXCEPTION_TESTS = []
TESTS = []

np.random.RandomState(123)


for p in TEST_NDARRAYS:
    EXCEPTION_TESTS.append(
        [
            {"keys": "img", "softmax": True, "sigmoid": False, "threshold": None, "remove_small_objects": True, "min_size": 10},
            p(np.random.rand(1, 5, 5, 5)),
            ValueError
        ]
    )

    EXCEPTION_TESTS.append(
        [
            {"keys": "img", "softmax": False, "sigmoid": True, "remove_small_objects": True, "min_size": 10},
            p(np.random.rand(1, 5, 5)),
            ValueError
        ]
    )

for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {"keys": "img", "mask_key_postfix": "mask", "softmax": True, "sigmoid": False, "threshold": None, "remove_small_objects": False, "min_size": 10},
            p([
                [[0.5022, 0.3403, 0.9997], [0.8793, 0.5514, 0.2697], [0.6134, 0.6389, 0.0680]],
                [[0.5000, 0.3400, 0.9900], [0.8900, 0.5600, 0.2700], [0.6100, 0.6300, 0.0600]],
              ]),
            (1, 3, 3),
            [0, 1],
        ]
    )

    TESTS.append(
        [
            {"keys": "img", "mask_key_postfix": "mask", "softmax": False, "sigmoid": True, "threshold": 0.5, "remove_small_objects": False, "min_size": 10},
            p([[[0.5022, 0.3403, 0.9997], [0.8793, 0.5514, 0.2697], [-0.1134, -0.0389, -0.0680]]]),
            (1, 3, 3),
            [0, 1]
        ]
    )




class TestGenerateMask(unittest.TestCase):
    @parameterized.expand(EXCEPTION_TESTS)
    def test_value(self, argments, image, exception_type):
        with self.assertRaises(exception_type):
            GenerateMaskd(**argments)({'img': image})

    @parameterized.expand(TESTS)
    def test_value2(self, argments, image, expected_shape, expected_value):
        result = GenerateMaskd(**argments)({'img': image})
        self.assertEqual(result['img_mask'].shape, expected_shape)

        if isinstance(result['img_mask'], torch.Tensor):
            result['img_mask'] = result['img_mask'].cpu().numpy()
        self.assertEqual(np.unique(result['img_mask']).tolist(), expected_value)


if __name__ == "__main__":
    unittest.main()

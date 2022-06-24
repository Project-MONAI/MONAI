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

import numpy as np
from parameterized import parameterized

from monai.transforms import CenterSpatialCropd
from tests.utils import TEST_NDARRAYS_ALL, assert_allclose

TEST_SHAPES = []
for p in TEST_NDARRAYS_ALL:
    TEST_SHAPES.append(
        [{"keys": "img", "roi_size": [2, -1, -1]}, {"img": p(np.random.randint(0, 2, size=[3, 3, 3, 3]))}, (3, 2, 3, 3)]
    )

    TEST_SHAPES.append(
        [{"keys": "img", "roi_size": [2, 2, 2]}, {"img": p(np.random.randint(0, 2, size=[3, 3, 3, 3]))}, (3, 2, 2, 2)]
    )

TEST_CASES = []
for p in TEST_NDARRAYS_ALL:
    TEST_CASES.append(
        [
            {"keys": "img", "roi_size": [2, 2]},
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[1, 2], [2, 3]]])),
        ]
    )


class TestCenterSpatialCropd(unittest.TestCase):
    @parameterized.expand(TEST_SHAPES)
    def test_shape(self, input_param, input_data, expected_shape):
        result = CenterSpatialCropd(**input_param)(input_data)
        self.assertTupleEqual(result["img"].shape, expected_shape)

    @parameterized.expand(TEST_CASES)
    def test_value(self, input_param, input_data, expected_value):
        result = CenterSpatialCropd(**input_param)(input_data)
        assert_allclose(result["img"], expected_value, type_test=False)


if __name__ == "__main__":
    unittest.main()

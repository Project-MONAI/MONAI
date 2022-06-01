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

from monai.data.meta_tensor import MetaTensor
from monai.transforms import RandSpatialCropSamplesd
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = [
    [
        {"keys": ["img", "seg"], "num_samples": 4, "roi_size": [2, 2, 2], "random_center": True},
        {"img": np.arange(81).reshape(3, 3, 3, 3), "seg": np.arange(81, 0, -1).reshape(3, 3, 3, 3)},
        [(3, 3, 3, 2), (3, 2, 2, 2), (3, 3, 3, 2), (3, 3, 2, 2)],
        {
            "img": np.array(
                [
                    [[[0, 1], [3, 4]], [[9, 10], [12, 13]], [[18, 19], [21, 22]]],
                    [[[27, 28], [30, 31]], [[36, 37], [39, 40]], [[45, 46], [48, 49]]],
                    [[[54, 55], [57, 58]], [[63, 64], [66, 67]], [[72, 73], [75, 76]]],
                ]
            ),
            "seg": np.array(
                [
                    [[[81, 80], [78, 77]], [[72, 71], [69, 68]], [[63, 62], [60, 59]]],
                    [[[54, 53], [51, 50]], [[45, 44], [42, 41]], [[36, 35], [33, 32]]],
                    [[[27, 26], [24, 23]], [[18, 17], [15, 14]], [[9, 8], [6, 5]]],
                ]
            ),
        },
    ],
    [
        {"keys": ["img", "seg"], "num_samples": 8, "roi_size": [2, 2, 3], "random_center": False},
        {"img": np.arange(81).reshape(3, 3, 3, 3), "seg": np.arange(81, 0, -1).reshape(3, 3, 3, 3)},
        [
            (3, 3, 3, 3),
            (3, 2, 3, 3),
            (3, 2, 2, 3),
            (3, 2, 3, 3),
            (3, 3, 3, 3),
            (3, 3, 3, 3),
            (3, 2, 2, 3),
            (3, 3, 2, 3),
        ],
        {
            "img": np.array(
                [
                    [[[0, 1, 2], [3, 4, 5]], [[9, 10, 11], [12, 13, 14]], [[18, 19, 20], [21, 22, 23]]],
                    [[[27, 28, 29], [30, 31, 32]], [[36, 37, 38], [39, 40, 41]], [[45, 46, 47], [48, 49, 50]]],
                    [[[54, 55, 56], [57, 58, 59]], [[63, 64, 65], [66, 67, 68]], [[72, 73, 74], [75, 76, 77]]],
                ]
            ),
            "seg": np.array(
                [
                    [[[81, 80, 79], [78, 77, 76]], [[72, 71, 70], [69, 68, 67]], [[63, 62, 61], [60, 59, 58]]],
                    [[[54, 53, 52], [51, 50, 49]], [[45, 44, 43], [42, 41, 40]], [[36, 35, 34], [33, 32, 31]]],
                    [[[27, 26, 25], [24, 23, 22]], [[18, 17, 16], [15, 14, 13]], [[9, 8, 7], [6, 5, 4]]],
                ]
            ),
        },
    ],
]


class TestRandSpatialCropSamplesd(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_shape(self, input_param, input_data, expected_shape, expected_last):
        for input_type in TEST_NDARRAYS:
            input_data_mod = {k: input_type(v) for k, v in input_data.items()}
            xform = RandSpatialCropSamplesd(**input_param)
            xform.set_random_state(1234)
            result = xform(input_data_mod)
            for i, (item, expected) in enumerate(zip(result, expected_shape)):
                for k in xform.keys:
                    v = item[k]
                    self.assertIsInstance(v, MetaTensor)
                    self.assertTupleEqual(v.shape, expected)
                    self.assertEqual(v.meta["patch_index"], i)

            assert_allclose(item["img"], expected_last["img"], type_test=False)
            assert_allclose(item["seg"], expected_last["seg"], type_test=False)

            with self.assertRaises(NotImplementedError):
                _ = xform.inverse(result)

            inv = xform.inverse(item)
            for k, v in inv.items():
                self.assertIsInstance(v, MetaTensor)
                self.assertTupleEqual(v.shape, input_data[k].shape)

    def test_deep_copy(self):
        data = {"img": np.ones((1, 10, 11, 12))}
        num_samples = 3
        sampler = RandSpatialCropSamplesd(
            keys=["img"], roi_size=(3, 3, 3), num_samples=num_samples, random_center=True, random_size=False
        )
        samples = sampler(data)
        self.assertEqual(len(samples), num_samples)
        for sample in samples:
            self.assertEqual(len(sample["img"].applied_operations), 1)


if __name__ == "__main__":
    unittest.main()

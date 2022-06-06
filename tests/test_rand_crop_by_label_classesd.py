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
from copy import deepcopy

import numpy as np
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from monai.transforms import ClassesToIndicesd, RandCropByLabelClassesd
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            # One-Hot label
            {
                "keys": "img",
                "label_key": "label",
                "num_classes": None,
                "spatial_size": [2, 2, -1],
                "ratios": [1, 1, 1],
                "num_samples": 2,
                "image_key": "image",
                "image_threshold": 0,
            },
            {
                "img": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "image": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "label": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
            },
            (3, 2, 2, 3),
        ]
    )

    TESTS.append(
        [
            # Argmax label
            {
                "keys": "img",
                "label_key": "label",
                "num_classes": 2,
                "spatial_size": [2, 2, 2],
                "ratios": [1, 1],
                "num_samples": 2,
                "image_key": "image",
                "image_threshold": 0,
            },
            {
                "img": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "image": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "label": p(np.random.randint(0, 2, size=[1, 3, 3, 3])),
            },
            (3, 2, 2, 2),
        ]
    )

    TESTS.append(
        [
            # Argmax label
            {
                "keys": "img",
                "label_key": "label",
                "num_classes": 2,
                "spatial_size": [4, 4, 2],
                "ratios": [1, 1],
                "num_samples": 2,
                "image_key": "image",
                "image_threshold": 0,
                "allow_smaller": True,
            },
            {
                "img": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "image": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "label": p(np.random.randint(0, 2, size=[1, 3, 3, 3])),
            },
            (3, 3, 3, 2),
        ]
    )

    TESTS.append(
        [
            # Argmax label
            {
                "keys": "img",
                "label_key": "label",
                "num_classes": 2,
                "spatial_size": [4, 4, 4],
                "ratios": [1, 1],
                "num_samples": 2,
                "image_key": "image",
                "image_threshold": 0,
                "allow_smaller": True,
            },
            {
                "img": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "image": p(np.random.randint(0, 2, size=[3, 3, 3, 3])),
                "label": p(np.random.randint(0, 2, size=[1, 3, 3, 3])),
            },
            (3, 3, 3, 3),
        ]
    )


class TestRandCropByLabelClassesd(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_type_shape(self, input_param, input_data, expected_shape):
        result = RandCropByLabelClassesd(**input_param)(input_data)
        self.assertIsInstance(result, list)
        # test with pre-computed indices
        input_data = ClassesToIndicesd(keys="label", num_classes=input_param["num_classes"])(input_data)
        input_param["indices_key"] = "label_cls_indices"
        cropper = RandCropByLabelClassesd(**input_param)
        result = cropper(input_data)
        self.assertIsInstance(result, list)
        for r in result:
            for k in cropper.keys:
                im = r[k]
                self.assertIsInstance(im, MetaTensor)
                self.assertEqual(len(im.applied_operations), 1)
                self.assertTupleEqual(im.shape, expected_shape)
            # individual inverse
            inv = cropper.inverse(deepcopy(r))
            for k in cropper.keys:
                im = inv[k]
                self.assertIsInstance(im, MetaTensor)
                self.assertEqual(im.applied_operations, [])
                self.assertTupleEqual(im.shape, input_data[k].shape)
        with self.assertRaises(NotImplementedError):
            _ = cropper.inverse(result)


if __name__ == "__main__":
    unittest.main()

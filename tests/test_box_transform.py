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

from monai.apps.detection.transforms.dictionary import ConvertBoxToStandardModed
from monai.transforms import Invertd
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = []

boxes = [[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]
image_size = [1, 5, 6, 4]
image = np.zeros(image_size)

for p in TEST_NDARRAYS:
    TESTS.append(
        [
            {"box_keys": "boxes", "mode": "xyzwhd"},
            {"boxes": p(boxes), "image": p(image)},
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 3, 3], [0, 1, 1, 2, 3, 4]]),
            p([[0, 1, 0, 2, 3, 3], [0, 1, 1, 2, 3, 4]]),
            p([[0, 1, 1, 2, 3, 4], [0, 1, 0, 2, 3, 3]]),
        ]
    )


class TestBoxTransform(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, keys, data, expected_standard_result, expected_clip_result, expected_flip_result):
        transform_convert_mode = ConvertBoxToStandardModed(**keys)
        result = transform_convert_mode(data)
        assert_allclose(result["boxes"], expected_standard_result, type_test=True, device_test=True, atol=0.0)

        invert_transform_convert_mode = Invertd(keys=["boxes"], transform=transform_convert_mode, orig_keys=["boxes"])
        data_back = invert_transform_convert_mode(result)
        assert_allclose(data_back["boxes"], data["boxes"], type_test=False, device_test=False, atol=0.0)


if __name__ == "__main__":
    unittest.main()

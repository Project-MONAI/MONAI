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

import numpy as np
from parameterized import parameterized

from monai.transforms import BoxClipToImaged, BoxConvertToStandardd, BoxFlipd

# this script test box_convert_mode, box_convert_standard_mode, and box_area
TESTS = []
bbox = [[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]
image_size = [1, 5, 6, 4]
image = np.zeros(image_size)
TESTS.append(
    [
        {"box_keys": "bbox", "box_mode": "xyzwhd"},
        {"bbox": np.array(bbox), "image": image},
        [[0, 0, 0, 0, 0, 0], [0, 2, 1, 3, 0, 3], [0, 2, 1, 3, 1, 4]],
        [[0, 2, 1, 3, 0, 3], [0, 2, 1, 3, 1, 4]],
        [[0, 2, 1, 3, 1, 4], [0, 2, 1, 3, 0, 3]],
    ]
)


class TestBoxTransform(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, keys, data, expected_standard_result, expected_clip_result, expected_flip_result):
        result = BoxConvertToStandardd(**keys)(data)
        np.testing.assert_equal(result["bbox"], expected_standard_result)

        result = BoxClipToImaged(box_keys="bbox", image_key="image")(result)
        np.testing.assert_equal(result["bbox"], expected_clip_result)

        result = BoxFlipd(box_keys="bbox", image_key="image", spatial_axis=2)(result)
        np.testing.assert_equal(result["bbox"], expected_flip_result)


if __name__ == "__main__":
    unittest.main()

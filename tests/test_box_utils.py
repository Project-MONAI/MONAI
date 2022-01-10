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

import torch
from parameterized import parameterized

from monai.data.box_utils import box_area, box_clip_to_image, box_convert_mode, box_convert_standard_mode
from tests.utils import TEST_NDARRAYS

# this script test box_convert_mode, box_convert_standard_mode, and box_area
TESTS = []
for p in TEST_NDARRAYS:
    bbox = [[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]
    image_size = [4, 4, 4]
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xyzwhd"},
            "xyzwhd",
            [[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]],
            [0, 12, 12],
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xyzwhd"},
            "xyzxyz",
            [[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 3, 3], [0, 1, 1, 2, 3, 4]],
            [0, 12, 12],
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xyzwhd"},
            "xxyyzz",
            [[0, 0, 0, 0, 0, 0], [0, 2, 1, 3, 0, 3], [0, 2, 1, 3, 1, 4]],
            [0, 12, 12],
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xyzxyz"},
            "xyzwhd",
            [[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 1, 3], [0, 1, 1, 2, 1, 2]],
            [0, 6, 4],
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xyzxyz"},
            "xyzxyz",
            [[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]],
            [0, 6, 4],
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xyzxyz"},
            "xxyyzz",
            [[0, 0, 0, 0, 0, 0], [0, 2, 1, 2, 0, 3], [0, 2, 1, 2, 1, 3]],
            [0, 6, 4],
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xxyyzz"},
            "xxyyzz",
            [[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]],
            [0, 2, 1],
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xxyyzz"},
            "xyzxyz",
            [[0, 0, 0, 0, 0, 0], [0, 0, 2, 1, 2, 3], [0, 1, 2, 1, 2, 3]],
            [0, 2, 1],
        ]
    )
    TESTS.append(
        [
            {"bbox": p(bbox), "image_size": image_size, "mode": "xxyyzz"},
            "xyzwhd",
            [[0, 0, 0, 0, 0, 0], [0, 0, 2, 1, 2, 1], [0, 1, 2, 1, 1, 1]],
            [0, 2, 1],
        ]
    )


class TestCreateBoxList(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, input_data, mode2, expected_box, expected_area):
        bbox1 = torch.tensor(input_data["bbox"])
        expected_box = torch.tensor(expected_box)
        mode1 = input_data["mode"]
        image_size = input_data["image_size"]

        result2 = box_convert_mode(bbox1=bbox1, mode1=mode1, mode2=mode2)
        self.assertEqual(result2.cpu().numpy().tolist(), expected_box.cpu().numpy().tolist())

        result1 = box_convert_mode(bbox1=result2, mode1=mode2, mode2=mode1)
        self.assertEqual(result1.cpu().numpy().tolist(), bbox1.cpu().numpy().tolist())

        result_standard = box_convert_standard_mode(bbox=bbox1, mode=mode1)
        expected_box_standard = box_convert_standard_mode(bbox=expected_box, mode=mode2)
        self.assertEqual(result_standard.cpu().numpy().tolist(), expected_box_standard.cpu().numpy().tolist())

        self.assertEqual(box_area(result_standard).tolist(), expected_area)

        result_standard_clip = box_clip_to_image(result_standard, image_size, mode=None, remove_empty=True)
        self.assertEqual(box_area(result_standard_clip).tolist(), list(filter(lambda num: num > 0, expected_area)))


if __name__ == "__main__":
    unittest.main()

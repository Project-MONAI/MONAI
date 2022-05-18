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

from monai.data.box_utils import (
    CenterSizeMode,
    CornerCornerModeTypeA,
    CornerCornerModeTypeB,
    CornerCornerModeTypeC,
    CornerSizeMode,
    convert_box_mode,
    convert_box_to_standard_mode,
)
from monai.utils.type_conversion import convert_data_type
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    boxes = [[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]
    spatial_size = [4, 4, 4]
    TESTS.append(
        [
            {"boxes": p(boxes), "spatial_size": spatial_size, "mode": "cccwhd", "half": False},
            CornerSizeMode,
            p([[0, 0, 0, 0, 0, 0], [-1, 0, -1.5, 2, 2, 3], [-1, 0, -0.5, 2, 2, 3]]),
            p([0, 12, 12]),
        ]
    )
    TESTS.append(
        [
            {"boxes": p(boxes), "spatial_size": spatial_size, "mode": "xyzwhd", "half": False},
            CornerSizeMode,
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]),
            p([0, 12, 12]),
        ]
    )
    TESTS.append(
        [
            {"boxes": p(boxes), "spatial_size": spatial_size, "mode": "xyzwhd", "half": True},
            "xyzxyz",
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 3, 3], [0, 1, 1, 2, 3, 4]]),
            p([0, 12, 12]),
        ]
    )
    TESTS.append(
        [
            {"boxes": p(boxes), "spatial_size": spatial_size, "mode": "xyzwhd", "half": False},
            "xxyyzz",
            p([[0, 0, 0, 0, 0, 0], [0, 2, 1, 3, 0, 3], [0, 2, 1, 3, 1, 4]]),
            p([0, 12, 12]),
        ]
    )
    TESTS.append(
        [
            {"boxes": p(boxes), "spatial_size": spatial_size, "mode": "xyzwhd", "half": False},
            CornerCornerModeTypeC,
            p([[0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 0, 3], [0, 1, 2, 3, 1, 4]]),
            p([0, 12, 12]),
        ]
    )
    TESTS.append(
        [
            {"boxes": p(boxes), "spatial_size": spatial_size, "mode": CornerCornerModeTypeA(), "half": False},
            "xyzwhd",
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 1, 3], [0, 1, 1, 2, 1, 2]]),
            p([0, 6, 4]),
        ]
    )
    TESTS.append(
        [
            {"boxes": p(boxes), "spatial_size": spatial_size, "mode": CornerCornerModeTypeA, "half": True},
            CornerCornerModeTypeA,
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]),
            p([0, 6, 4]),
        ]
    )
    TESTS.append(
        [
            {"boxes": p(boxes), "spatial_size": spatial_size, "mode": "xyzxyz", "half": False},
            CornerCornerModeTypeB(),
            p([[0, 0, 0, 0, 0, 0], [0, 2, 1, 2, 0, 3], [0, 2, 1, 2, 1, 3]]),
            p([0, 6, 4]),
        ]
    )
    TESTS.append(
        [
            {"boxes": p(boxes), "spatial_size": spatial_size, "mode": "xxyyzz", "half": False},
            "xxyyzz",
            p([[0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 2, 3], [0, 1, 1, 2, 2, 3]]),
            p([0, 2, 1]),
        ]
    )
    TESTS.append(
        [
            {"boxes": p(boxes), "spatial_size": spatial_size, "mode": "xxyyzz", "half": True},
            "xyzxyz",
            p([[0, 0, 0, 0, 0, 0], [0, 0, 2, 1, 2, 3], [0, 1, 2, 1, 2, 3]]),
            p([0, 2, 1]),
        ]
    )
    TESTS.append(
        [
            {"boxes": p(boxes), "spatial_size": spatial_size, "mode": "xxyyzz", "half": False},
            "xyzwhd",
            p([[0, 0, 0, 0, 0, 0], [0, 0, 2, 1, 2, 1], [0, 1, 2, 1, 1, 1]]),
            p([0, 2, 1]),
        ]
    )
    TESTS.append(
        [
            {"boxes": p(boxes), "spatial_size": spatial_size, "mode": "xxyyzz", "half": False},
            CenterSizeMode(),
            p([[0, 0, 0, 0, 0, 0], [0.5, 1, 2.5, 1, 2, 1], [0.5, 1.5, 2.5, 1, 1, 1]]),
            p([0, 2, 1]),
        ]
    )


class TestCreateBoxList(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, input_data, mode2, expected_box, expected_area):
        expected_box = convert_data_type(expected_box, dtype=np.float32)[0]
        boxes1 = convert_data_type(input_data["boxes"], dtype=np.float32)[0]
        mode1 = input_data["mode"]
        half_bool = input_data["half"]

        # test float16
        if half_bool:
            boxes1 = convert_data_type(boxes1, dtype=np.float16)[0]
            expected_box = convert_data_type(expected_box, dtype=np.float16)[0]

        # test convert_box_mode, convert_box_to_standard_mode
        result2 = convert_box_mode(boxes=boxes1, src_mode=mode1, dst_mode=mode2)
        assert_allclose(result2, expected_box, type_test=True, device_test=True, atol=0.0)

        result1 = convert_box_mode(boxes=result2, src_mode=mode2, dst_mode=mode1)
        assert_allclose(result1, boxes1, type_test=True, device_test=True, atol=0.0)

        result_standard = convert_box_to_standard_mode(boxes=boxes1, mode=mode1)
        expected_box_standard = convert_box_to_standard_mode(boxes=expected_box, mode=mode2)
        assert_allclose(result_standard, expected_box_standard, type_test=True, device_test=True, atol=0.0)


if __name__ == "__main__":
    unittest.main()

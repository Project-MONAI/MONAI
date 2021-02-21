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

from monai.apps.deepgrow.transforms import (
    AddGuidanceSignald,
    AddInitialSeedPointd,
    AddRandomGuidanced,
    FindAllValidSlicesd,
    FindDiscrepancyRegionsd,
    SpatialCropForegroundd,
)

IMAGE = np.array([[[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]]])
LABEL = np.array([[[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]])
BATCH_IMAGE = np.array([[[[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]]]])
BATCH_LABEL = np.array([[[[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]]])

DATA_1 = {
    "image": IMAGE,
    "label": LABEL,
    "image_meta_dict": {},
    "label_meta_dict": {},
}

DATA_2 = {
    "image": np.array(
        [
            [
                [[1, 2, 3, 2, 1], [1, 1, 3, 2, 1], [0, 0, 0, 0, 0], [1, 1, 1, 2, 1], [0, 2, 2, 2, 1]],
                [[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]],
            ]
        ]
    ),
    "label": np.array(
        [
            [
                [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]],
            ]
        ]
    ),
    "guidance": np.array([[[1, 0, 2, 2], [1, 1, 2, 2]], [[-1, -1, -1, -1], [-1, -1, -1, -1]]]),
}

DATA_3 = {
    "image": BATCH_IMAGE,
    "label": BATCH_LABEL,
    "pred": np.array([[[[[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]]]),
}

DATA_4 = {
    "image": BATCH_IMAGE,
    "label": BATCH_LABEL,
    "guidance": np.array([[[[1, 0, 2, 2]], [[-1, -1, -1, -1]]]]),
    "discrepancy": np.array(
        [
            [
                [[[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
                [[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
            ]
        ]
    ),
    "probability": [1.0],
}

FIND_SLICE_TEST_CASE_1 = [
    {"label": "label", "sids": "sids"},
    DATA_1,
    [0],
]

FIND_SLICE_TEST_CASE_2 = [
    {"label": "label", "sids": "sids"},
    DATA_2,
    [0, 1],
]

CROP_TEST_CASE_1 = [
    {
        "keys": ["image", "label"],
        "source_key": "label",
        "select_fn": lambda x: x > 0,
        "channel_indices": None,
        "margin": 0,
        "spatial_size": [1, 4, 4],
    },
    DATA_1,
    np.array([[[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]]),
]

ADD_INITIAL_POINT_TEST_CASE_1 = [
    {"label": "label", "guidance": "guidance", "sids": "sids"},
    DATA_1,
    np.array([[[1, 0, 2, 2]], [[-1, -1, -1, -1]]]),
]

ADD_GUIDANCE_TEST_CASE_1 = [
    {"image": "image", "guidance": "guidance", "batched": False},
    DATA_2,
    np.array(
        [
            [
                [[1, 2, 3, 2, 1], [1, 1, 3, 2, 1], [0, 0, 0, 0, 0], [1, 1, 1, 2, 1], [0, 2, 2, 2, 1]],
                [[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]],
            ],
            [
                [
                    [0.0, 0.26689214, 0.37996644, 0.26689214, 0.0],
                    [0.26689214, 0.65222847, 0.81548417, 0.65222847, 0.26689214],
                    [0.37996635, 0.81548399, 1.0, 0.81548399, 0.37996635],
                    [0.26689214, 0.65222847, 0.81548417, 0.65222847, 0.26689214],
                    [0.0, 0.26689214, 0.37996644, 0.26689214, 0.0],
                ],
                [
                    [0.0, 0.26689214, 0.37996644, 0.26689214, 0.0],
                    [0.26689214, 0.65222847, 0.81548417, 0.65222847, 0.26689214],
                    [0.37996635, 0.81548399, 1.0, 0.81548399, 0.37996635],
                    [0.26689214, 0.65222847, 0.81548417, 0.65222847, 0.26689214],
                    [0.0, 0.26689214, 0.37996644, 0.26689214, 0.0],
                ],
            ],
            [
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            ],
        ]
    ),
]

FIND_DISCREPANCY_TEST_CASE_1 = [
    {"label": "label", "pred": "pred", "discrepancy": "discrepancy"},
    DATA_3,
    np.array(
        [
            [
                [[[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
                [[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
            ]
        ]
    ),
]

ADD_RANDOM_GUIDANCE_TEST_CASE_1 = [
    {"guidance": "guidance", "discrepancy": "discrepancy", "probability": "probability", "batched": True},
    DATA_4,
    np.array([[[[1, 0, 2, 2], [1, 0, 1, 3]], [[-1, -1, -1, -1], [-1, -1, -1, -1]]]]),
]


class TestFindAllValidSlicesd(unittest.TestCase):
    @parameterized.expand([FIND_SLICE_TEST_CASE_1, FIND_SLICE_TEST_CASE_2])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = FindAllValidSlicesd(**arguments)(input_data)
        np.testing.assert_allclose(result[arguments["sids"]], expected_result)


class TestSpatialCropForegroundd(unittest.TestCase):
    @parameterized.expand([CROP_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = SpatialCropForegroundd(**arguments)(input_data)
        np.testing.assert_allclose(result["image"], expected_result)

    @parameterized.expand([CROP_TEST_CASE_1])
    def test_foreground_position(self, arguments, input_data, _):
        result = SpatialCropForegroundd(**arguments)(input_data)
        np.testing.assert_allclose(result["image_meta_dict"]["foreground_start_coord"], np.array([0, 1, 1]))
        np.testing.assert_allclose(result["image_meta_dict"]["foreground_end_coord"], np.array([1, 4, 4]))

        arguments["start_coord_key"] = "test_start_coord"
        arguments["end_coord_key"] = "test_end_coord"
        result = SpatialCropForegroundd(**arguments)(input_data)
        np.testing.assert_allclose(result["image_meta_dict"]["test_start_coord"], np.array([0, 1, 1]))
        np.testing.assert_allclose(result["image_meta_dict"]["test_end_coord"], np.array([1, 4, 4]))


class TestAddInitialSeedPointd(unittest.TestCase):
    @parameterized.expand([ADD_INITIAL_POINT_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        seed = 0
        add_fn = AddInitialSeedPointd(**arguments)
        add_fn.set_random_state(seed)
        result = add_fn(input_data)
        np.testing.assert_allclose(result[arguments["guidance"]], expected_result)


class TestAddGuidanceSignald(unittest.TestCase):
    @parameterized.expand([ADD_GUIDANCE_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = AddGuidanceSignald(**arguments)(input_data)
        np.testing.assert_allclose(result["image"], expected_result, rtol=1e-5)


class TestFindDiscrepancyRegionsd(unittest.TestCase):
    @parameterized.expand([FIND_DISCREPANCY_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = FindDiscrepancyRegionsd(**arguments)(input_data)
        np.testing.assert_allclose(result[arguments["discrepancy"]], expected_result)


class TestAddRandomGuidanced(unittest.TestCase):
    @parameterized.expand([ADD_RANDOM_GUIDANCE_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        seed = 0
        add_fn = AddRandomGuidanced(**arguments)
        add_fn.set_random_state(seed)
        result = add_fn(input_data)
        np.testing.assert_allclose(result[arguments["guidance"]], expected_result, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

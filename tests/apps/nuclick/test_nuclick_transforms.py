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

import numpy as np
from parameterized import parameterized

from monai.apps.nuclick.transforms import (
    AddClickSignalsd,
    AddLabelAsGuidanced,
    AddPointGuidanceSignald,
    ExtractPatchd,
    FilterImaged,
    FlattenLabeld,
    PostFilterLabeld,
    SetLabelClassd,
    SplitLabeld,
)

# Data Definitions
RGB_IMAGE_1 = np.array(
    [[[0, 0, 0], [0, 1, 0], [0, 0, 1]], [[2, 0, 2], [0, 1, 0], [1, 0, 1]], [[3, 0, 2], [0, 1, 0], [1, 3, 1]]]
)

LABEL_1 = np.array(
    [
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1],
    ],
    dtype=np.uint8,
)

LABEL_1_1 = np.array(
    [
        [1, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 2, 2, 2],
        [1, 1, 1, 0, 2, 2, 2],
        [1, 1, 1, 0, 2, 2, 2],
    ],
    dtype=np.uint8,
)

LABEL_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)

LABEL_3 = np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]], dtype=np.uint8)

LABEL_4 = np.array([[[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]]], dtype=np.uint8)

# target nucleus (1), a 9-pixel neighbor (2) and a 1-pixel fragment of another neighbor (3)
LABEL_5 = np.array(
    [
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 2],
            [0, 0, 0, 2, 2, 2],
            [3, 0, 0, 2, 2, 2],
        ]
    ],
    dtype=np.uint8,
)

IL_IMAGE_1 = np.array(
    [
        [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1]],
        [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1]],
        [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1]],
    ]
)

IL_FG_IMAGE_1 = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1]])

IL_LABEL_1 = np.array(
    [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]], dtype=np.uint8
)

IL_OTHERS_1 = np.array(
    [[[1, 1, 1, 1, 1], [2, 0, 0, 0, 2], [3, 0, 0, 0, 3], [4, 0, 0, 0, 4], [5, 5, 5, 5, 5]]], dtype=np.uint8
)

IL_IMAGE_2 = np.array(
    [[[0, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [0, 0, 1]]]
)

IL_LABEL_2 = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=np.uint8)

# two 16-pixel blobs in one 10x10 patch
PRED_2 = np.zeros((1, 10, 10), dtype=np.float32)
PRED_2[0, 1:5, 1:5] = 1.0
PRED_2[0, 6:10, 6:10] = 1.0

# click inside the first blob
NUC_POINTS_2 = np.zeros((1, 1, 10, 10), dtype=np.float32)
NUC_POINTS_2[0, 0, 2, 2] = 1.0

# click outside both blobs
NUC_POINTS_3 = np.zeros((1, 1, 10, 10), dtype=np.float32)
NUC_POINTS_3[0, 0, 0, 9] = 1.0

PRED_1 = np.array(
    [[[1, 1, 1, 1, 1], [2, 0, 0, 0, 2], [3, 0, 0, 0, 3], [4, 0, 0, 0, 4], [5, 5, 5, 5, 5]]], dtype=np.float32
)

NUC_POINTS_1 = np.array(
    [
        [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]],
        [[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
    ],
    dtype=np.float32,
)
BB_1 = np.array([[0, 0, 5, 5], [1, 1, 6, 6]], dtype=np.uint8)

DATA_FILTER_1 = {"image": RGB_IMAGE_1}

DATA_FLATTEN_1 = {"label": LABEL_1}
DATA_FLATTEN_2 = {"label": LABEL_2}

DATA_EXTRACT_1 = {"image": IL_IMAGE_1, "label": IL_LABEL_1, "centroid": (2, 2)}
DATA_EXTRACT_2 = {"image": IL_IMAGE_2, "label": IL_LABEL_2, "centroid": (1, 1)}

DATA_SPLIT_1 = {"label": LABEL_3, "mask_value": 1}
DATA_SPLIT_2 = {"label": LABEL_4, "mask_value": 4}
DATA_SPLIT_3 = {"label": LABEL_5, "mask_value": 1}

DATA_GUIDANCE_1 = {"image": IL_IMAGE_1, "label": IL_LABEL_1, "others": IL_OTHERS_1, "centroid": (2, 2)}

DATA_CLICK_1 = {"image": IL_IMAGE_1, "foreground": [[2, 2], [1, 1]]}

DATA_LABEL_FILTER_1 = {
    "pred": PRED_1,
    "nuc_points": NUC_POINTS_1,
    "bounding_boxes": BB_1,
    "img_height": 6,
    "img_width": 6,
}

DATA_LABEL_FILTER_2 = {
    "pred": PRED_2,
    "nuc_points": NUC_POINTS_2,
    "bounding_boxes": [[0, 0, 10, 10]],
    "img_height": 10,
    "img_width": 10,
}

# Result Definitions
EXTRACT_RESULT_TC1 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 1]]], dtype=np.uint8)
EXTRACT_RESULT_TC2 = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=np.uint8)

SPLIT_RESULT_TC1 = np.array([[[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=np.uint8)
SPLIT_RESULT_TC2 = np.array([[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]], dtype=np.uint8)

SPLIT_OTHERS_ALL = np.array(
    [
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [2, 0, 0, 1, 1, 1],
        ]
    ],
    dtype=np.uint8,
)
SPLIT_OTHERS_NO_FRAGMENT = SPLIT_OTHERS_ALL * (SPLIT_OTHERS_ALL == 1)

LABEL_FILTER_BOTH_BLOBS = (PRED_2[0] > 0).astype(np.uint8)
LABEL_FILTER_CLICKED_BLOB = np.zeros((10, 10), dtype=np.uint8)
LABEL_FILTER_CLICKED_BLOB[1:5, 1:5] = 1

# Test Case Definitions
FILTER_IMAGE_TEST_CASE_1 = [{"keys": "image", "min_size": 1}, DATA_FILTER_1, [3, 3, 3]]

FLATTEN_LABEL_TEST_CASE_1 = [{"keys": "label"}, DATA_FLATTEN_1, [0, 1, 2, 3]]
FLATTEN_LABEL_TEST_CASE_2 = [{"keys": "label"}, DATA_FLATTEN_2, [0]]
FLATTEN_LABEL_TEST_CASE_3 = [{"keys": "label"}, {"label": LABEL_1_1}, [0, 1, 2, 3, 4]]

EXTRACT_TEST_CASE_1 = [{"keys": ["image", "label"], "patch_size": 3}, DATA_EXTRACT_1, [1, 3, 3]]
EXTRACT_TEST_CASE_2 = [{"keys": ["image", "label"], "patch_size": 5}, DATA_EXTRACT_1, [1, 5, 5]]
EXTRACT_TEST_CASE_3 = [{"keys": ["image", "label"], "patch_size": 1}, DATA_EXTRACT_2, [1, 1, 1]]

EXTRACT_RESULT_TEST_CASE_1 = [{"keys": ["image", "label"], "patch_size": 3}, DATA_EXTRACT_1, EXTRACT_RESULT_TC1]
EXTRACT_RESULT_TEST_CASE_2 = [{"keys": ["image", "label"], "patch_size": 4}, DATA_EXTRACT_2, EXTRACT_RESULT_TC2]

EXTRACT_KW_TEST_CASE_1 = [
    {"keys": ["image", "label"], "patch_size": 3, "mode": "constant"},
    DATA_EXTRACT_1,
    EXTRACT_RESULT_TC1,
]

SPLIT_TEST_CASE_1 = [{"keys": ["label"], "mask_value": "mask_value", "min_area": 1}, DATA_SPLIT_1, SPLIT_RESULT_TC1]
SPLIT_TEST_CASE_2 = [{"keys": ["label"], "mask_value": "mask_value", "min_area": 3}, DATA_SPLIT_2, SPLIT_RESULT_TC2]

SPLIT_OTHERS_TEST_CASE_1 = [{"keys": ["label"]}, DATA_SPLIT_3, SPLIT_OTHERS_ALL]
SPLIT_OTHERS_TEST_CASE_2 = [{"keys": ["label"], "min_area": 1}, DATA_SPLIT_3, SPLIT_OTHERS_ALL]
SPLIT_OTHERS_TEST_CASE_3 = [{"keys": ["label"], "min_area": 2}, DATA_SPLIT_3, SPLIT_OTHERS_NO_FRAGMENT]
SPLIT_OTHERS_TEST_CASE_4 = [{"keys": ["label"], "min_area": 9}, DATA_SPLIT_3, SPLIT_OTHERS_NO_FRAGMENT]
SPLIT_OTHERS_TEST_CASE_5 = [{"keys": ["label"], "min_area": 10}, DATA_SPLIT_3, np.zeros_like(SPLIT_OTHERS_ALL)]

GUIDANCE_TEST_CASE_1 = [{"image": "image", "label": "label", "others": "others"}, DATA_GUIDANCE_1, [5, 5, 5]]
GUIDANCE_TEST_CASE_2 = [
    {"image": "image", "label": "label", "others": "others", "gaussian": True, "use_distance": True},
    DATA_GUIDANCE_1,
    [5, 5, 5],
]

CLICK_TEST_CASE_1 = [{"image": "image", "foreground": "foreground", "bb_size": 4}, DATA_CLICK_1, [2, 5, 4, 4]]
CLICK_TEST_CASE_2 = [
    {"image": "image", "foreground": "foreground", "bb_size": 4, "gaussian": True},
    DATA_CLICK_1,
    [2, 5, 4, 4],
]

LABEL_FILTER_TEST_CASE_1 = [{"keys": ["pred"]}, DATA_LABEL_FILTER_1, [6, 6]]

LABEL_FILTER_RESULT_TEST_CASE_1 = [{"keys": ["pred"]}, DATA_LABEL_FILTER_2, LABEL_FILTER_BOTH_BLOBS]
LABEL_FILTER_RESULT_TEST_CASE_2 = [
    {"keys": ["pred"], "do_reconstruction": True},
    DATA_LABEL_FILTER_2,
    LABEL_FILTER_CLICKED_BLOB,
]
LABEL_FILTER_RESULT_TEST_CASE_3 = [
    {"keys": ["pred"], "do_reconstruction": True},
    {**DATA_LABEL_FILTER_2, "nuc_points": NUC_POINTS_2[:, 0]},
    LABEL_FILTER_CLICKED_BLOB,
]

LABEL_GUIDANCE_TEST_CASE_1 = [{"keys": ["image"], "source": "label"}, DATA_GUIDANCE_1, [4, 5, 5]]

LABEL_CLASS_TEST_CASE_1 = [{"keys": ["label"], "offset": 2}, DATA_GUIDANCE_1, 3]

# Test Case Classes


class TestFilterImaged(unittest.TestCase):

    @parameterized.expand([FILTER_IMAGE_TEST_CASE_1])
    def test_correct_shape(self, arguments, input_data, expected_shape):
        result = FilterImaged(**arguments)(input_data)
        np.testing.assert_equal(result["image"].shape, expected_shape)


class TestFlattenLabeld(unittest.TestCase):

    @parameterized.expand([FLATTEN_LABEL_TEST_CASE_1, FLATTEN_LABEL_TEST_CASE_2, FLATTEN_LABEL_TEST_CASE_3])
    def test_correct_num_labels(self, arguments, input_data, expected_result):
        result = FlattenLabeld(**arguments)(input_data)
        np.testing.assert_equal(np.unique(result["label"]), expected_result)


class TestExtractPatchd(unittest.TestCase):

    @parameterized.expand([EXTRACT_TEST_CASE_1, EXTRACT_TEST_CASE_2, EXTRACT_TEST_CASE_3])
    def test_correct_patch_size(self, arguments, input_data, expected_shape):
        result = ExtractPatchd(**arguments)(input_data)
        np.testing.assert_equal(result["label"].shape, expected_shape)

    @parameterized.expand([EXTRACT_RESULT_TEST_CASE_1, EXTRACT_RESULT_TEST_CASE_2, EXTRACT_KW_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = ExtractPatchd(**arguments)(input_data)
        np.testing.assert_equal(result["label"], expected_result)


class TestSplitLabelsd(unittest.TestCase):

    @parameterized.expand([SPLIT_TEST_CASE_1, SPLIT_TEST_CASE_2])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = SplitLabeld(**arguments)(input_data)
        np.testing.assert_equal(result["label"], expected_result)

    @parameterized.expand(
        [
            SPLIT_OTHERS_TEST_CASE_1,
            SPLIT_OTHERS_TEST_CASE_2,
            SPLIT_OTHERS_TEST_CASE_3,
            SPLIT_OTHERS_TEST_CASE_4,
            SPLIT_OTHERS_TEST_CASE_5,
        ]
    )
    def test_min_area(self, arguments, input_data, expected_others):
        result = SplitLabeld(**arguments)(input_data)
        np.testing.assert_equal(result["label"], (LABEL_5 == 1).astype(np.uint8))
        np.testing.assert_equal(result["others"], expected_others)


class TestGuidanceSignal(unittest.TestCase):

    @parameterized.expand([GUIDANCE_TEST_CASE_1, GUIDANCE_TEST_CASE_2])
    def test_correct_shape(self, arguments, input_data, expected_shape):
        result = AddPointGuidanceSignald(**arguments)(input_data)
        np.testing.assert_equal(result["image"].shape, expected_shape)


class TestClickSignal(unittest.TestCase):

    @parameterized.expand([CLICK_TEST_CASE_1, CLICK_TEST_CASE_2])
    def test_correct_shape(self, arguments, input_data, expected_shape):
        result = AddClickSignalsd(**arguments)(input_data)
        np.testing.assert_equal(result["image"].shape, expected_shape)


class TestPostFilterLabel(unittest.TestCase):

    @parameterized.expand([LABEL_FILTER_TEST_CASE_1])
    def test_correct_shape(self, arguments, input_data, expected_shape):
        result = PostFilterLabeld(**arguments)(input_data)
        np.testing.assert_equal(result["pred"].shape, expected_shape)

    @parameterized.expand(
        [LABEL_FILTER_RESULT_TEST_CASE_1, LABEL_FILTER_RESULT_TEST_CASE_2, LABEL_FILTER_RESULT_TEST_CASE_3]
    )
    def test_correct_results(self, arguments, input_data, expected_result):
        result = PostFilterLabeld(**arguments)(input_data)
        np.testing.assert_equal(result["pred"], expected_result)

    def test_reconstruction_click_outside_mask(self):
        data = {**DATA_LABEL_FILTER_2, "nuc_points": NUC_POINTS_3}
        with self.assertWarns(UserWarning):
            result = PostFilterLabeld(keys=["pred"], do_reconstruction=True)(data)
        np.testing.assert_equal(result["pred"], LABEL_FILTER_BOTH_BLOBS)


class TestAddLabelAsGuidance(unittest.TestCase):

    @parameterized.expand([LABEL_GUIDANCE_TEST_CASE_1])
    def test_correct_shape(self, arguments, input_data, expected_shape):
        result = AddLabelAsGuidanced(**arguments)(input_data)
        np.testing.assert_equal(result["image"].shape, expected_shape)


class TestSetLabelClass(unittest.TestCase):

    @parameterized.expand([LABEL_CLASS_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = SetLabelClassd(**arguments)(input_data)
        assert result["label"] == expected_result


if __name__ == "__main__":
    unittest.main()

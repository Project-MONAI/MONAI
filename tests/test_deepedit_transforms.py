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

from monai.apps.deepedit.transforms import (
    AddGuidanceFromPointsCustomd,
    AddGuidanceSignalCustomd,
    AddInitialSeedPointMissingLabelsd,
    AddRandomGuidanceCustomd,
    DiscardAddGuidanced,
    FindAllValidSlicesMissingLabelsd,
    FindDiscrepancyRegionsCustomd,
    NormalizeLabelsInDatasetd,
    ResizeGuidanceMultipleLabelCustomd,
    SingleLabelSelectiond,
    SplitPredsLabeld,
    ToCheckTransformd,
)
from monai.utils.enums import PostFix

IMAGE = np.array([[[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]]])
LABEL = np.array([[[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]])
MULTI_LABEL = np.random.randint(0, 6, (5, 5))
PRED = np.random.randint(0, 6, (5, 5))
LABEL_NAMES = {"spleen": 1, "right kidney": 2, "background": 0}

DATA_1 = {
    "image": IMAGE,
    "label": LABEL,
    PostFix.meta("image"): {"dim": IMAGE.shape},
    PostFix.meta("label"): {},
    "foreground": [0, 0, 0],
    "background": [0, 0, 0],
}

DATA_2 = {
    "image": IMAGE,
    "label": LABEL,
    "guidance": np.array([[[1, 0, 2, 2]], [[-1, -1, -1, -1]]]),
    "discrepancy": np.array(
        [
            [[[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
            [[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
        ]
    ),
    "probability": 1.0,
}

DATA_3 = {
    "image": np.arange(1000).reshape((1, 5, 10, 20)),
    PostFix.meta("image"): {"foreground_cropped_shape": (1, 10, 20, 40), "dim": [3, 512, 512, 128]},
    "guidance": [[[6, 10, 14], [8, 10, 14]], [[8, 10, 16]]],
    "foreground": [[10, 14, 6], [10, 14, 8]],
    "background": [[10, 16, 8]],
}

DATA_5 = {
    "image": np.arange(25).reshape((1, 5, 5)),
    PostFix.meta("image"): {"spatial_shape": [5, 5, 1]},
    "foreground": [[2, 2, 0]],
    "background": [],
}

DATA_6 = {
    "image": IMAGE,
    "label": MULTI_LABEL,
    "guidance": {
        "spleen": np.array([[1, 0, 2, 2], [-1, -1, -1, -1]]),
        "right kidney": np.array([[1, 0, 2, 2], [-1, -1, -1, -1]]),
        "background": np.array([[1, 0, 2, 2], [-1, -1, -1, -1]]),
    },
    "probability": 1.0,
    "label_names": LABEL_NAMES,
    "pred": PRED,
}

DATA_7 = {
    "image": IMAGE,
    "label": MULTI_LABEL,
    "current_label": "spleen",
    "probability": 1.0,
    "label_names": LABEL_NAMES,
    "pred": PRED,
}

ADD_GUIDANCE_FROM_POINTS_TEST_CASE_1 = [
    {"ref_image": "image", "dimensions": 3, "guidance": "guidance", "depth_first": True},
    DATA_5,
    [[0, 2, 2]],
    [],
]

ADD_GUIDANCE_FROM_POINTS_TEST_CASE_2 = [
    {"ref_image": "image", "dimensions": 3, "guidance": "guidance", "depth_first": True},
    DATA_6,
    [[0, 2, 2]],
    [[0, 1, 0]],
]

ADD_GUIDANCE_FROM_POINTS_TEST_CASE_3 = [
    {"ref_image": "image", "dimensions": 3, "guidance": "guidance", "depth_first": True},
    DATA_7,
    [[3, 5, 7], [4, 5, 7]],
    [[4, 5, 8]],
]

ADD_GUIDANCE_CUSTOM_TEST_CASE = [DATA_6, 4]

ADD_INITIAL_POINT_TEST_CASE_1 = [
    {"label": "label", "guidance": "guidance", "sids": "sids"},
    DATA_1,
    "[[[1, 0, 2, 2]], [[-1, -1, -1, -1]]]",
]

CLICK_RATIO_ADD_RANDOM_GUIDANCE_TEST_CASE_1 = [
    {"guidance": "guidance", "discrepancy": "discrepancy", "probability": "probability"},
    DATA_2,
    "[[[1, 0, 2, 2], [-1, -1, -1, -1]], [[-1, -1, -1, -1], [1, 0, 2, 1]]]",
]

DISCARD_ADD_GUIDANCE_TEST_CASE = [{"image": IMAGE, "label": LABEL}, DATA_1, (3, 1, 5, 5)]

FIND_DISCREPANCY_TEST_CASE_1 = [
    {"label": "label", "pred": "pred", "discrepancy": "discrepancy"},
    DATA_3,
    np.array(
        [
            [[[[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
            [[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]],
        ]
    ),
]

FIND_SLICE_TEST_CASE_1 = [{"label": "label", "sids": "sids"}, DATA_1, [0]]

FIND_SLICE_TEST_CASE_2 = [{"label": "label", "sids": "sids"}, DATA_2, [0, 1]]

NormalizeLabelsDatasetd_TEST_CASE = [{"label_names": LABEL_NAMES}, DATA_6, len(LABEL_NAMES)]

RESIZE_GUIDANCE_TEST_CASE_1 = [
    {"ref_image": "image", "guidance": "guidance"},
    DATA_3,
    [[[0, 0, 0], [0, 0, 1]], [[0, 0, 1]]],
]

SingleLabelSelectiond_TEST_CASE = [{"label_names": ["spleen"]}, DATA_7, "spleen"]

SplitPredsLabeld_TEST_CASE = [DATA_7]

ToCheckTransformd_TEST_CASE = [DATA_7, 6]


class TestAddGuidanceFromPointsCustomd(unittest.TestCase):
    @parameterized.expand(
        [
            ADD_GUIDANCE_FROM_POINTS_TEST_CASE_1,
            ADD_GUIDANCE_FROM_POINTS_TEST_CASE_2,
            ADD_GUIDANCE_FROM_POINTS_TEST_CASE_3,
        ]
    )
    def test_correct_results(self, arguments, input_data, expected_pos, expected_neg):
        result = AddGuidanceFromPointsCustomd(**arguments)(input_data)
        self.assertEqual(result[arguments["guidance"]][0], expected_pos)
        self.assertEqual(result[arguments["guidance"]][1], expected_neg)


class TestAddGuidanceSignalCustomd(unittest.TestCase):
    @parameterized.expand([ADD_GUIDANCE_CUSTOM_TEST_CASE])
    def test_correct_results(self, input_data, expected_result):
        add_fn = AddGuidanceSignalCustomd(keys="image")
        result = add_fn(input_data)
        self.assertEqual(result["image"].shape[0], expected_result)


class TestAddInitialSeedPointMissingLabelsd(unittest.TestCase):
    @parameterized.expand([ADD_INITIAL_POINT_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        seed = 0
        add_fn = AddInitialSeedPointMissingLabelsd(**arguments)
        add_fn.set_random_state(seed)
        result = add_fn(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class TestAddRandomGuidanceCustomd(unittest.TestCase):
    @parameterized.expand([CLICK_RATIO_ADD_RANDOM_GUIDANCE_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        seed = 0
        add_fn = AddRandomGuidanceCustomd(**arguments)
        add_fn.set_random_state(seed)
        result = add_fn(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class TestDiscardAddGuidanced(unittest.TestCase):
    @parameterized.expand([DISCARD_ADD_GUIDANCE_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = DiscardAddGuidanced(arguments)
        result = add_fn(input_data)
        self.assertEqual(result["image"].shape, expected_result)


class TestFindAllValidSlicesMissingLabelsd(unittest.TestCase):
    @parameterized.expand([FIND_SLICE_TEST_CASE_1, FIND_SLICE_TEST_CASE_2])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = FindAllValidSlicesMissingLabelsd(**arguments)(input_data)
        np.testing.assert_allclose(result[arguments["sids"]], expected_result)


class TestFindDiscrepancyRegionsCustomd(unittest.TestCase):
    @parameterized.expand([FIND_DISCREPANCY_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = FindDiscrepancyRegionsCustomd(**arguments)(input_data)
        np.testing.assert_allclose(result[arguments["discrepancy"]], expected_result)


class TestNormalizeLabelsDatasetd(unittest.TestCase):
    @parameterized.expand([NormalizeLabelsDatasetd_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = NormalizeLabelsInDatasetd(keys="label", **arguments)
        result = add_fn(input_data)
        self.assertEqual(len(np.unique(result["label"])), expected_result)


class TestResizeGuidanceMultipleLabelCustomd(unittest.TestCase):
    @parameterized.expand([RESIZE_GUIDANCE_TEST_CASE_1])
    def test_correct_results(self, arguments, input_data, expected_result):
        result = ResizeGuidanceMultipleLabelCustomd(**arguments)(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class TestSingleLabelSelectiond(unittest.TestCase):
    @parameterized.expand([SingleLabelSelectiond_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = SingleLabelSelectiond(keys="label", **arguments)
        result = add_fn(input_data)
        self.assertEqual(result["current_label"], expected_result)


class TestSplitPredsLabeld(unittest.TestCase):
    @parameterized.expand([SplitPredsLabeld_TEST_CASE])
    def test_correct_results(self, input_data):
        add_fn = SplitPredsLabeld(keys="pred")
        result = add_fn(input_data)
        self.assertIsNotNone(result["pred_spleen"])


class TestToCheckTransformd(unittest.TestCase):
    @parameterized.expand([ToCheckTransformd_TEST_CASE])
    def test_correct_results(self, input_data, expected_result):
        add_fn = ToCheckTransformd(keys="label")
        result = add_fn(input_data)
        self.assertEqual(len(result), expected_result)


if __name__ == "__main__":
    unittest.main()

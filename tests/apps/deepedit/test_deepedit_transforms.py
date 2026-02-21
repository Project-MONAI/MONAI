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

from monai.apps.deepedit.transforms import (
    AddGuidanceFromPointsDeepEditd,
    AddGuidanceSignalDeepEditd,
    AddInitialSeedPointMissingLabelsd,
    AddRandomGuidanceDeepEditd,
    DiscardAddGuidanced,
    FindAllValidSlicesMissingLabelsd,
    FindDiscrepancyRegionsDeepEditd,
    NormalizeLabelsInDatasetd,
    RemapLabelsToSequentiald,
    ResizeGuidanceMultipleLabelDeepEditd,
    SingleLabelSelectiond,
    SplitPredsLabeld,
)
from monai.utils import min_version, optional_import, set_determinism
from monai.utils.enums import PostFix

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

set_determinism(seed=0)
IMAGE = np.random.randint(0, 256, size=(1, 10, 10, 10))
THREE_CHAN_IMAGE = np.random.randint(0, 255, size=(3, 10, 10, 10))
LABEL = np.random.randint(0, 2, size=(10, 10, 10))
PRED = np.random.randint(0, 2, size=(10, 10, 10))
LABEL_NAMES = {"spleen": 1, "background": 0}
DISCREPANCY = {
    "spleen": np.random.randint(0, 2, size=(10, 10, 10)),
    "background": np.random.randint(0, 2, size=(10, 10, 10)),
}
set_determinism(None)

DATA_1 = {
    "image": IMAGE,
    "label": LABEL,
    PostFix.meta("image"): {"dim": IMAGE.shape, "spatial_shape": IMAGE[0, ...].shape},
    PostFix.meta("label"): {},
}

DATA_2 = {
    "image": IMAGE,
    "label": LABEL,
    "label_names": LABEL_NAMES,
    "guidance": {"spleen": [[3, 5, 4, 6], [-1, -1, -1, -1]], "background": [[-1, -1, -1, -1], [-1, -1, -1, -1]]},
    "discrepancy": DISCREPANCY,
    "probability": 1.0,
}

DATA_3 = {
    "image": IMAGE,
    "label": LABEL,
    "guidance": {
        "spleen": np.array([[1, 0, 2, 2], [-1, -1, -1, -1]]),
        "background": np.array([[1, 0, 2, 2], [-1, -1, -1, -1]]),
    },
    "probability": 1.0,
    "label_names": LABEL_NAMES,
    "pred": PRED,
}

DATA_4 = {
    "image": IMAGE,
    "label": LABEL,
    PostFix.meta("image"): {"dim": IMAGE.shape, "spatial_shape": IMAGE[0, ...].shape},
    "current_label": "spleen",
    "probability": 1.0,
    "label_names": LABEL_NAMES,
    "spleen": [[0, 4, 3], [0, 0, 3], [0, 1, 3]],
    "sids": {"spleen": []},
    "pred": PRED,
}

DATA_5 = {
    "image": IMAGE,
    "label": LABEL,
    PostFix.meta("image"): {"dim": IMAGE.shape, "spatial_shape": IMAGE[0, ...].shape},
    "current_label": "spleen",
    "probability": 1.0,
    "label_names": LABEL_NAMES,
    "sids": {"spleen": [2, 3, 4], "background": [0, 1, 5]},
}

DATA_6 = {
    "image": IMAGE,
    "label": LABEL[None],
    PostFix.meta("image"): {"dim": IMAGE.shape, "spatial_shape": IMAGE[0, ...].shape},
    "current_label": "spleen",
    "label_names": LABEL_NAMES,
}

DATA_7 = {
    "image": IMAGE,
    "label": LABEL,
    "pred": PRED,
    PostFix.meta("image"): {"dim": IMAGE.shape, "spatial_shape": IMAGE[0, ...].shape},
    "current_label": "spleen",
    "probability": 1.0,
    "label_names": LABEL_NAMES,
    "guidance": {
        "spleen": np.array([[1, 0, 2, 2], [-1, -1, -1, -1]]),
        "background": np.array([[1, 0, 2, 2], [-1, -1, -1, -1]]),
    },
}

DATA_8 = {
    "image": IMAGE,
    "label": LABEL,
    PostFix.meta("image"): {"dim": IMAGE.shape, "spatial_shape": IMAGE[0, ...].shape},
    "label_names": LABEL_NAMES,
}

DATA_9 = {
    "image": IMAGE,
    "label": LABEL,
    PostFix.meta("image"): {"dim": IMAGE.shape, "spatial_shape": IMAGE[0, ...].shape},
    "label_names": LABEL_NAMES,
    "guidance": {"spleen": np.array([0, 2, 2]), "background": np.array([-1, -1, -1])},
}

DATA_10 = {
    "image": IMAGE,
    "label": LABEL,
    PostFix.meta("image"): {"dim": IMAGE.shape, "spatial_shape": IMAGE[0, ...].shape},
    "current_label": "spleen",
}

DATA_11 = {"image": IMAGE, "label": LABEL, "label_names": LABEL_NAMES, "pred": PRED}

ADD_GUIDANCE_FROM_POINTS_TEST_CASE = [
    {"ref_image": "image", "guidance": "guidance", "label_names": LABEL_NAMES},  # arguments
    DATA_4,  # input_data
    [0, 4, 3],  # expected_result
]

ADD_GUIDANCE_CUSTOM_TEST_CASE = [
    {"keys": "image", "guidance": "guidance"},  # arguments
    DATA_3,  # input_data
    3,  # expected_result
]

ADD_INITIAL_POINT_TEST_CASE = [
    {"keys": "label", "guidance": "guidance", "sids": "sids"},  # arguments
    DATA_5,  # input_data
    {
        "spleen": "[[1, 0, 7], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]",
        "background": "[[1, 5, 3], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]",
    },  # expected_result
]

ADD_RANDOM_GUIDANCE_TEST_CASE = [
    {"keys": "NA", "guidance": "guidance", "discrepancy": "discrepancy", "probability": "probability"},  # arguments
    DATA_2,  # input_data
    0,  # expected_result
]

DISCARD_ADD_GUIDANCE_TEST_CASE = [
    {"keys": "image", "label_names": LABEL_NAMES},  # arguments
    DATA_1,  # input_data
    (3, 10, 10, 10),  # expected_result
]

FIND_DISCREPANCY_TEST_CASE = [
    {"keys": "label", "pred": "pred", "discrepancy": "discrepancy"},  # arguments
    DATA_7,  # input_data
    240,  # expected_result
]

FIND_SLICE_TEST_CASE = [
    {"keys": "label", "sids": "sids"},  # arguments
    DATA_6,  # input_data
    {"spleen": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "background": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},  # expected_result
]

NormalizeLabelsDatasetd_TEST_CASE = [
    {"keys": "label", "label_names": LABEL_NAMES},  # arguments
    DATA_8,  # input_data
    len(LABEL_NAMES),  # expected_result
]

RESIZE_GUIDANCE_TEST_CASE = [
    {"guidance": "guidance", "ref_image": "image"},  # arguments
    DATA_9,  # input_data
    {"spleen": [0, 2, 2], "background": [-1, -1, -1]},  # expected_result
]

SingleLabelSelectiond_TEST_CASE = [
    {"keys": "label", "label_names": ["spleen"]},  # arguments
    DATA_10,  # input_data
    "spleen",  # expected_result
]

SplitPredsLabeld_TEST_CASE = [{"keys": "pred"}, DATA_11, (1, 10, 10)]  # arguments  # input_data  # expected_result


class TestAddGuidanceFromPointsCustomd(unittest.TestCase):

    @parameterized.expand([ADD_GUIDANCE_FROM_POINTS_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = AddGuidanceFromPointsDeepEditd(**arguments)
        result = add_fn(input_data)
        self.assertEqual(result[arguments["guidance"]]["spleen"][0], expected_result)


class TestAddGuidanceSignalCustomd(unittest.TestCase):

    @parameterized.expand([ADD_GUIDANCE_CUSTOM_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = AddGuidanceSignalDeepEditd(**arguments)
        result = add_fn(input_data)
        self.assertEqual(result["image"].shape[0], expected_result)


class TestAddInitialSeedPointMissingLabelsd(unittest.TestCase):

    @parameterized.expand([ADD_INITIAL_POINT_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        seed = 0
        add_fn = AddInitialSeedPointMissingLabelsd(**arguments)
        add_fn.set_random_state(seed)
        result = add_fn(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class TestAddRandomGuidanceCustomd(unittest.TestCase):

    @parameterized.expand([ADD_RANDOM_GUIDANCE_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = AddRandomGuidanceDeepEditd(**arguments)
        result = add_fn(input_data)
        label_key = list(result[arguments["guidance"]].keys())[0]
        self.assertGreaterEqual(len(result[arguments["guidance"]][label_key]), expected_result)


class TestDiscardAddGuidanced(unittest.TestCase):

    @parameterized.expand([DISCARD_ADD_GUIDANCE_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = DiscardAddGuidanced(**arguments)
        result = add_fn(input_data)
        self.assertEqual(result["image"].shape, expected_result)


class TestFindAllValidSlicesMissingLabelsd(unittest.TestCase):

    @parameterized.expand([FIND_SLICE_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = FindAllValidSlicesMissingLabelsd(**arguments)
        result = add_fn(input_data)
        self.assertEqual(result[arguments["sids"]], expected_result)


class TestFindDiscrepancyRegionsCustomd(unittest.TestCase):

    @parameterized.expand([FIND_DISCREPANCY_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = FindDiscrepancyRegionsDeepEditd(**arguments)
        result = add_fn(input_data)
        self.assertEqual(np.sum(result[arguments["discrepancy"]]["spleen"][0]), expected_result)


class TestNormalizeLabelsDatasetd(unittest.TestCase):

    @parameterized.expand([NormalizeLabelsDatasetd_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = NormalizeLabelsInDatasetd(**arguments)
        result = add_fn(input_data)
        self.assertEqual(len(np.unique(result["label"])), expected_result)

    def test_ordering_determinism(self):
        """Test that different input ordering produces the same output (alphabetical)"""
        # Create a label array with different label values
        label = np.array([[[0, 1, 6, 3]]])  # background=0, spleen=1, liver=6, kidney=3

        # Test case 1: liver first, then kidney, then spleen
        data1 = {"label": label.copy()}
        transform1 = RemapLabelsToSequentiald(
            keys="label", label_names={"liver": 6, "kidney": 3, "spleen": 1, "background": 0}
        )
        result1 = transform1(data1)

        # Test case 2: spleen first, then kidney, then liver (different order)
        data2 = {"label": label.copy()}
        transform2 = RemapLabelsToSequentiald(
            keys="label", label_names={"spleen": 1, "kidney": 3, "liver": 6, "background": 0}
        )
        result2 = transform2(data2)

        # Both should produce the same output (alphabetically sorted)
        # Expected mapping: background=0, kidney=1, liver=2, spleen=3
        np.testing.assert_array_equal(result1["label"], result2["label"])

        # Verify the actual mapping is alphabetical
        expected_output = np.array([[[0, 3, 2, 1]]])  # kidney=1, liver=2, spleen=3, background=0
        np.testing.assert_array_equal(result1["label"], expected_output)

        # Verify label_names is correct
        self.assertEqual(result1["label_names"], {"background": 0, "kidney": 1, "liver": 2, "spleen": 3})
        self.assertEqual(result2["label_names"], {"background": 0, "kidney": 1, "liver": 2, "spleen": 3})

    def test_multiple_labels(self):
        """Test with multiple non-background labels"""
        label = np.array([[[0, 1, 2, 5]]])  # background, spleen, kidney, liver
        data = {"label": label.copy()}
        transform = RemapLabelsToSequentiald(
            keys="label", label_names={"spleen": 1, "kidney": 2, "liver": 5, "background": 0}
        )
        result = transform(data)

        # Expected: background=0, kidney=1, liver=2, spleen=3 (alphabetical)
        expected = np.array([[[0, 3, 1, 2]]])
        np.testing.assert_array_equal(result["label"], expected)
        self.assertEqual(result["label_names"], {"background": 0, "kidney": 1, "liver": 2, "spleen": 3})

    def test_deprecated_name_warning(self):
        """Test that NormalizeLabelsInDatasetd is properly deprecated.

        The deprecation warning only triggers when MONAI version >= 1.6 (since="1.6").
        This test verifies:
        1. The actual NormalizeLabelsInDatasetd class is marked as deprecated in docstring
        2. The class is a subclass of RemapLabelsToSequentiald
        3. The deprecation mechanism works correctly (tested via version_val simulation)
        4. The actual class functions correctly
        """
        import warnings

        from monai.utils import deprecated

        # Verify NormalizeLabelsInDatasetd docstring indicates deprecation
        self.assertIn("deprecated", NormalizeLabelsInDatasetd.__doc__.lower())
        self.assertIn("RemapLabelsToSequentiald", NormalizeLabelsInDatasetd.__doc__)

        # Verify NormalizeLabelsInDatasetd is a subclass of RemapLabelsToSequentiald
        self.assertTrue(issubclass(NormalizeLabelsInDatasetd, RemapLabelsToSequentiald))

        # Test the deprecation mechanism using version_val to simulate version 1.6
        # This verifies the @deprecated decorator behavior that NormalizeLabelsInDatasetd uses
        @deprecated(
            since="1.6",
            removed="1.8",
            msg_suffix="Use `RemapLabelsToSequentiald` instead.",
            version_val="1.6",  # Simulate version 1.6 to trigger warning
        )
        class DeprecatedNormalizeLabels(RemapLabelsToSequentiald):
            pass

        data = {"label": np.array([[[0, 1]]])}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transform = DeprecatedNormalizeLabels(keys="label", label_names={"spleen": 1, "background": 0})
            _ = transform(data)

            # Check that a deprecation warning was raised
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, FutureWarning))
            self.assertIn("RemapLabelsToSequentiald", str(w[0].message))

        # Verify the actual NormalizeLabelsInDatasetd class works correctly
        transform_actual = NormalizeLabelsInDatasetd(keys="label", label_names={"spleen": 1, "background": 0})
        result = transform_actual({"label": np.array([[[0, 1]]])})
        self.assertIn("label", result)


class TestResizeGuidanceMultipleLabelCustomd(unittest.TestCase):

    @parameterized.expand([RESIZE_GUIDANCE_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = ResizeGuidanceMultipleLabelDeepEditd(**arguments)
        result = add_fn(input_data)
        self.assertEqual(result[arguments["guidance"]], expected_result)


class TestSingleLabelSelectiond(unittest.TestCase):

    @parameterized.expand([SingleLabelSelectiond_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = SingleLabelSelectiond(**arguments)
        result = add_fn(input_data)
        self.assertEqual(result["current_label"], expected_result)


class TestSplitPredsLabeld(unittest.TestCase):

    @parameterized.expand([SplitPredsLabeld_TEST_CASE])
    def test_correct_results(self, arguments, input_data, expected_result):
        add_fn = SplitPredsLabeld(**arguments)
        result = add_fn(input_data)
        self.assertEqual(result["pred_spleen"].shape, expected_result)


if __name__ == "__main__":
    unittest.main()

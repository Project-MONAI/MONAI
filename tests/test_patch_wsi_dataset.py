import os
import unittest
from unittest import skipUnless

import numpy as np
from numpy.testing import assert_array_equal
from parameterized import parameterized

from monai.apps.pathology.datasets import PatchWSIDataset
from monai.apps.utils import download_url
from monai.utils import optional_import

_, has_cim = optional_import("cucim")
_, has_osl = optional_import("openslide")

FILE_URL = "http://openslide.cs.cmu.edu/download/openslide-testdata/Generic-TIFF/CMU-1.tiff"
FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", os.path.basename(FILE_URL))

TEST_CASE_0 = [
    {
        "data": [
            {"image": FILE_PATH, "location": [0, 0], "label": [1]},
        ],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])},
    ],
]

TEST_CASE_1 = [
    {
        "data": [{"image": FILE_PATH, "location": [10004, 20004], "label": [0, 0, 0, 1]}],
        "region_size": (8, 8),
        "grid_shape": (2, 2),
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
    ],
]

TEST_CASE_2 = [
    {
        "data": [
            {"image": FILE_PATH, "location": [0, 0], "label": [1]},
        ],
        "region_size": 1,
        "grid_shape": 1,
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])},
    ],
]

TEST_CASE_3 = [
    {
        "data": [
            {"image": FILE_PATH, "location": [0, 0], "label": [[[0, 1], [1, 0]]]},
        ],
        "region_size": 1,
        "grid_shape": 1,
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 1], [1, 0]]])},
    ],
]

TEST_CASE_OPENSLIDE_0 = [
    {
        "data": [
            {"image": FILE_PATH, "location": [0, 0], "label": [1]},
        ],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "image_reader_name": "OpenSlide",
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])},
    ],
]

TEST_CASE_OPENSLIDE_1 = [
    {
        "data": [{"image": FILE_PATH, "location": [10004, 20004], "label": [0, 0, 0, 1]}],
        "region_size": (8, 8),
        "grid_shape": (2, 2),
        "patch_size": 1,
        "image_reader_name": "OpenSlide",
    },
    [
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
    ],
]


class TestPatchWSIDataset(unittest.TestCase):
    def setUp(self):
        download_url(FILE_URL, FILE_PATH, "5a3cfd4fd725c50578ddb80b517b759f")

    @parameterized.expand(
        [
            TEST_CASE_0,
            TEST_CASE_1,
            TEST_CASE_2,
            TEST_CASE_3,
        ]
    )
    @skipUnless(has_cim, "Requires CuCIM")
    def test_read_patches_cucim(self, input_parameters, expected):
        dataset = PatchWSIDataset(**input_parameters)
        samples = dataset[0]
        for i in range(len(samples)):
            self.assertTupleEqual(samples[i]["label"].shape, expected[i]["label"].shape)
            self.assertTupleEqual(samples[i]["image"].shape, expected[i]["image"].shape)
            self.assertIsNone(assert_array_equal(samples[i]["label"], expected[i]["label"]))
            self.assertIsNone(assert_array_equal(samples[i]["image"], expected[i]["image"]))

    @parameterized.expand(
        [
            TEST_CASE_OPENSLIDE_0,
            TEST_CASE_OPENSLIDE_1,
        ]
    )
    @skipUnless(has_osl, "Requires OpenSlide")
    def test_read_patches_openslide(self, input_parameters, expected):
        dataset = PatchWSIDataset(**input_parameters)
        samples = dataset[0]
        for i in range(len(samples)):
            self.assertTupleEqual(samples[i]["label"].shape, expected[i]["label"].shape)
            self.assertTupleEqual(samples[i]["image"].shape, expected[i]["image"].shape)
            self.assertIsNone(assert_array_equal(samples[i]["label"], expected[i]["label"]))
            self.assertIsNone(assert_array_equal(samples[i]["image"], expected[i]["image"]))


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from unittest import skipUnless
from urllib import request

import numpy as np
from numpy.testing import assert_array_equal
from parameterized import parameterized

from monai.apps.pathology.datasets import PatchWSIDataset
from monai.utils import optional_import

_, has_cim = optional_import("cucim")

FILE_URL = "http://openslide.cs.cmu.edu/download/openslide-testdata/Generic-TIFF/CMU-1.tiff"

TEST_CASE_0 = [
    FILE_URL,
    {
        "data": [
            {"image": "./CMU-1.tiff", "location": [0, 0], "label": [1]},
        ],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[1]])},
    ],
]

TEST_CASE_1 = [
    FILE_URL,
    {
        "data": [{"image": "./CMU-1.tiff", "location": [10000, 20000], "label": [0, 0, 0, 1]}],
        "region_size": (8, 8),
        "grid_shape": (2, 2),
        "patch_size": 1,
    },
    [
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[0]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[0]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[0]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[1]])},
    ],
]


class TestPatchWSIDataset(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1])
    @skipUnless(has_cim, "Requires CuCIM")
    def test_read_patches(self, file_url, input_parameters, expected):
        self.camelyon_data_download(file_url)
        dataset = PatchWSIDataset(**input_parameters)
        samples = dataset[0]
        for i in range(len(samples)):
            self.assertTupleEqual(samples[i]["label"].shape, expected[i]["label"].shape)
            self.assertTupleEqual(samples[i]["image"].shape, expected[i]["image"].shape)
            self.assertIsNone(assert_array_equal(samples[i]["label"], expected[i]["label"]))
            self.assertIsNone(assert_array_equal(samples[i]["image"], expected[i]["image"]))

    def camelyon_data_download(self, file_url):
        filename = os.path.basename(file_url)
        if not os.path.exists(filename):
            print(f"Test image [{filename}] does not exist. Downloading...")
            request.urlretrieve(file_url, filename)
        return filename


if __name__ == "__main__":
    unittest.main()

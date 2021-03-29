import os
import unittest
from unittest import skipUnless
from urllib import request

import numpy as np
from numpy.testing import assert_array_equal
from parameterized import parameterized

from monai.apps.pathology.datasets import SmartCachePatchWSIDataset
from monai.utils import optional_import

_, has_cim = optional_import("cucim")

FILE_URL = "http://openslide.cs.cmu.edu/download/openslide-testdata/Generic-TIFF/CMU-1.tiff"

TEST_CASE_0 = [
    FILE_URL,
    {
        "data": [
            {"image": "./CMU-1.tiff", "location": [0, 0], "label": [0]},
            {"image": "./CMU-1.tiff", "location": [0, 0], "label": [1]},
            {"image": "./CMU-1.tiff", "location": [0, 0], "label": [2]},
            {"image": "./CMU-1.tiff", "location": [0, 0], "label": [3]},
        ],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "transform": lambda x: x,
        "image_reader_name": "cuCIM",
        "replace_rate": 0.5,
        "cache_num": 2,
        "num_init_workers": 1,
        "num_replace_workers": 1,
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[3]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[3]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0]]])},
    ],
]

TEST_CASE_1 = [
    FILE_URL,
    {
        "data": [
            {"image": "./CMU-1.tiff", "location": [0, 0], "label": [[0, 0]]},
            {"image": "./CMU-1.tiff", "location": [0, 0], "label": [[1, 1]]},
            {"image": "./CMU-1.tiff", "location": [0, 0], "label": [[2, 2]]},
        ],
        "region_size": (1, 1),
        "grid_shape": (1, 1),
        "patch_size": 1,
        "transform": lambda x: x,
        "image_reader_name": "cuCIM",
        "replace_rate": 0.5,
        "cache_num": 2,
        "num_init_workers": 1,
        "num_replace_workers": 1,
    },
    [
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 0]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1, 1]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[1, 1]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[2, 2]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[2, 2]]])},
        {"image": np.array([[[239]], [[239]], [[239]]], dtype=np.uint8), "label": np.array([[[0, 0]]])},
    ],
]

TEST_CASE_2 = [
    FILE_URL,
    {
        "data": [
            {"image": "./CMU-1.tiff", "location": [10004, 20004], "label": [0, 0, 0, 0]},
            {"image": "./CMU-1.tiff", "location": [10004, 20004], "label": [1, 1, 1, 1]},
            {"image": "./CMU-1.tiff", "location": [10004, 20004], "label": [2, 2, 2, 2]},
        ],
        "region_size": (8, 8),
        "grid_shape": (2, 2),
        "patch_size": 1,
        "transform": lambda x: x,
        "image_reader_name": "cuCIM",
        "replace_rate": 0.5,
        "cache_num": 2,
        "num_init_workers": 1,
        "num_replace_workers": 1,
    },
    [
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[1]]])},
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[2]]])},
        {"image": np.array([[[247]], [[245]], [[248]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[245]], [[247]], [[244]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
        {"image": np.array([[[246]], [[246]], [[246]]], dtype=np.uint8), "label": np.array([[[0]]])},
    ],
]


class TestSmartCachePatchWSIDataset(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_0,
            TEST_CASE_1,
            TEST_CASE_2,
        ]
    )
    @skipUnless(has_cim, "Requires CuCIM")
    def test_read_patches(self, file_url, input_parameters, expected):
        self.camelyon_data_download(file_url)
        dataset = SmartCachePatchWSIDataset(**input_parameters)
        self.assertEqual(len(dataset), input_parameters["cache_num"])
        total_num_samples = len(input_parameters["data"])
        num_epochs = int(
            np.ceil(total_num_samples / (input_parameters["cache_num"] * input_parameters["replace_rate"]))
        )

        dataset.start()
        i = 0
        for _ in range(num_epochs):
            for j in range(len(dataset)):
                samples = dataset[j]
                n_patches = len(samples)
                self.assert_samples_expected(samples, expected[i : i + n_patches])
                i += n_patches
            dataset.update_cache()
        dataset.shutdown()

    def camelyon_data_download(self, file_url):
        filename = os.path.basename(file_url)
        if not os.path.exists(filename):
            print(f"Test image [{filename}] does not exist. Downloading...")
            request.urlretrieve(file_url, filename)
        return filename

    def assert_samples_expected(self, samples, expected):
        for i in range(len(samples)):
            self.assertTupleEqual(samples[i]["label"].shape, expected[i]["label"].shape)
            self.assertTupleEqual(samples[i]["image"].shape, expected[i]["image"].shape)
            self.assertIsNone(assert_array_equal(samples[i]["label"], expected[i]["label"]))
            self.assertIsNone(assert_array_equal(samples[i]["image"], expected[i]["image"]))


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from unittest import skipUnless
from urllib import request

import numpy as np
from numpy.testing import assert_array_equal
from parameterized import parameterized

from monai.apps.pathology.datasets import MaskedInferenceWSIDataset
from monai.utils import optional_import
from tests.utils import skip_if_quick

_, has_cim = optional_import("cucim")
_, has_osl = optional_import("openslide")

FILE_URL = "http://openslide.cs.cmu.edu/download/openslide-testdata/Generic-TIFF/CMU-1.tiff"

HEIGHT = 32914
WIDTH = 46000

mask = np.zeros((WIDTH // 2, HEIGHT // 2))
mask[100, 100] = 1
np.save("./tests/testing_data/mask1.npy", mask)
mask[100, 100:102] = 1
np.save("./tests/testing_data/mask2.npy", mask)
mask[100:102, 100:102] = 1
np.save("./tests/testing_data/mask4.npy", mask)

TEST_CASE_0 = [
    FILE_URL,
    {
        "data": [
            {"image": "./CMU-1.tiff", "label": "./tests/testing_data/mask1.npy"},
        ],
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [100, 100],
        },
    ],
]

TEST_CASE_1 = [
    FILE_URL,
    {
        "data": [{"image": "./CMU-1.tiff", "label": "./tests/testing_data/mask2.npy"}],
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [100, 100],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [101, 100],
        },
    ],
]

TEST_CASE_2 = [
    FILE_URL,
    {
        "data": [{"image": "./CMU-1.tiff", "label": "./tests/testing_data/mask4.npy"}],
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [100, 100],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [100, 101],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [101, 100],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [101, 101],
        },
    ],
]

TEST_CASE_3 = [
    FILE_URL,
    {
        "data": [
            {"image": "./CMU-1.tiff", "label": "./tests/testing_data/mask1.npy"},
        ],
        "patch_size": 2,
        "image_reader_name": "cuCIM",
    },
    [
        {
            "image": np.array(
                [
                    [[243, 243], [243, 243]],
                    [[243, 243], [243, 243]],
                    [[243, 243], [243, 243]],
                ],
                dtype=np.uint8,
            ),
            "name": "CMU-1",
            "mask_location": [100, 100],
        },
    ],
]

TEST_CASE_4 = [
    FILE_URL,
    {
        "data": [
            {"image": "./CMU-1.tiff", "label": "./tests/testing_data/mask1.npy"},
            {"image": "./CMU-1.tiff", "label": "./tests/testing_data/mask2.npy"},
        ],
        "patch_size": 1,
        "image_reader_name": "cuCIM",
    },
    [
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [100, 100],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [100, 100],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [101, 100],
        },
    ],
]


TEST_CASE_OPENSLIDE_0 = [
    FILE_URL,
    {
        "data": [
            {"image": "./CMU-1.tiff", "label": "./tests/testing_data/mask1.npy"},
        ],
        "patch_size": 1,
        "image_reader_name": "OpenSlide",
    },
    [
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [100, 100],
        },
    ],
]

TEST_CASE_OPENSLIDE_1 = [
    FILE_URL,
    {
        "data": [{"image": "./CMU-1.tiff", "label": "./tests/testing_data/mask2.npy"}],
        "patch_size": 1,
        "image_reader_name": "OpenSlide",
    },
    [
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [100, 100],
        },
        {
            "image": np.array([[[243]], [[243]], [[243]]], dtype=np.uint8),
            "name": "CMU-1",
            "mask_location": [101, 100],
        },
    ],
]


class TestMaskedInferenceWSIDataset(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_0,
            TEST_CASE_1,
            TEST_CASE_2,
            TEST_CASE_3,
            TEST_CASE_4,
        ]
    )
    @skipUnless(has_cim, "Requires CuCIM")
    @skip_if_quick
    def test_read_patches_cucim(self, file_url, input_parameters, expected):
        self.camelyon_data_download(file_url)
        dataset = MaskedInferenceWSIDataset(**input_parameters)
        self.compare_samples_expected(dataset, expected)

    @parameterized.expand(
        [
            TEST_CASE_OPENSLIDE_0,
            TEST_CASE_OPENSLIDE_1,
        ]
    )
    @skipUnless(has_osl, "Requires OpenSlide")
    @skip_if_quick
    def test_read_patches_openslide(self, file_url, input_parameters, expected):
        self.camelyon_data_download(file_url)
        dataset = MaskedInferenceWSIDataset(**input_parameters)
        self.compare_samples_expected(dataset, expected)

    def camelyon_data_download(self, file_url):
        filename = os.path.basename(file_url)
        if not os.path.exists(filename):
            print(f"Test image [{filename}] does not exist. Downloading...")
            request.urlretrieve(file_url, filename)
        return filename

    def compare_samples_expected(self, dataset, expected):
        for i in range(len(dataset)):
            self.assertTupleEqual(dataset[i][0]["image"].shape, expected[i]["image"].shape)
            self.assertIsNone(assert_array_equal(dataset[i][0]["image"], expected[i]["image"]))
            self.assertEqual(dataset[i][0]["name"], expected[i]["name"])
            self.assertListEqual(dataset[i][0]["mask_location"], expected[i]["mask_location"])


if __name__ == "__main__":
    unittest.main()

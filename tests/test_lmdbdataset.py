# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.data import LMDBDataset, json_hashing
from monai.transforms import Compose, LoadNiftid, SimulateDelayd, Transform
from tests.utils import skip_if_windows

TEST_CASE_1 = [
    Compose(
        [
            LoadNiftid(keys=["image", "label", "extra"]),
            SimulateDelayd(keys=["image", "label", "extra"], delay_time=[1e-7, 1e-6, 1e-5]),
        ]
    ),
    (128, 128, 128),
]

TEST_CASE_2 = [
    [
        LoadNiftid(keys=["image", "label", "extra"]),
        SimulateDelayd(keys=["image", "label", "extra"], delay_time=[1e-7, 1e-6, 1e-5]),
    ],
    (128, 128, 128),
]

TEST_CASE_3 = [None, (128, 128, 128), None]

TEST_CASE_4 = [
    [
        LoadNiftid(keys=["image", "label", "extra"]),
        SimulateDelayd(keys=["image", "label", "extra"], delay_time=[1e-7, 1e-6, 1e-5]),
    ],
    (128, 128, 128),
    {"db_name": "test42"},
]

TEST_CASE_5 = [
    [
        LoadNiftid(keys=["image", "label", "extra"]),
        SimulateDelayd(keys=["image", "label", "extra"], delay_time=[1e-7, 1e-6, 1e-5]),
    ],
    (128, 128, 128),
    {"pickle_protocol": 2, "lmdb_kwargs": {"map_size": 100 * 1024 ** 2}},
]

TEST_CASE_6 = [
    [
        LoadNiftid(keys=["image", "label", "extra"]),
        SimulateDelayd(keys=["image", "label", "extra"], delay_time=[1e-7, 1e-6, 1e-5]),
    ],
    (128, 128, 128),
    {"db_name": "testdb", "lmdb_kwargs": {"map_size": 100 * 1024 ** 2}},
]

TEST_CASE_7 = [
    [
        LoadNiftid(keys=["image", "label", "extra"]),
        SimulateDelayd(keys=["image", "label", "extra"], delay_time=[1e-7, 1e-6, 1e-5]),
    ],
    (128, 128, 128),
    {"db_name": "testdb", "lmdb_kwargs": {"map_size": 2 * 1024 ** 2}},
]


@skip_if_windows
class TestLMDBDataset(unittest.TestCase):
    def test_cache(self):
        """testing no inplace change to the hashed item"""
        items = [[list(range(i))] for i in range(5)]

        class _InplaceXform(Transform):
            def __call__(self, data):
                if data:
                    data[0] = data[0] + np.pi
                else:
                    data.append(1)
                return data

        with tempfile.TemporaryDirectory() as tempdir:
            ds = LMDBDataset(items, transform=_InplaceXform(), cache_dir=tempdir, lmdb_kwargs={"map_size": 10 * 1024})
            self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])
            ds1 = LMDBDataset(items, transform=_InplaceXform(), cache_dir=tempdir, lmdb_kwargs={"map_size": 10 * 1024})
            self.assertEqual(list(ds1), list(ds))
            self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])

            ds = LMDBDataset(
                items,
                transform=_InplaceXform(),
                cache_dir=tempdir,
                lmdb_kwargs={"map_size": 10 * 1024},
                hash_func=json_hashing,
            )
            self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])
            ds1 = LMDBDataset(
                items,
                transform=_InplaceXform(),
                cache_dir=tempdir,
                lmdb_kwargs={"map_size": 10 * 1024},
                hash_func=json_hashing,
            )
            self.assertEqual(list(ds1), list(ds))
            self.assertEqual(items, [[[]], [[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]]])

        self.assertTrue(isinstance(ds1.info(), dict))

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7])
    def test_shape(self, transform, expected_shape, kwargs=None):
        kwargs = kwargs or {}
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=[128, 128, 128]), np.eye(4))
        with tempfile.TemporaryDirectory() as tempdir:
            nib.save(test_image, os.path.join(tempdir, "test_image1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_label1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_extra1.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_image2.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_label2.nii.gz"))
            nib.save(test_image, os.path.join(tempdir, "test_extra2.nii.gz"))
            test_data = [
                {
                    "image": os.path.join(tempdir, "test_image1.nii.gz"),
                    "label": os.path.join(tempdir, "test_label1.nii.gz"),
                    "extra": os.path.join(tempdir, "test_extra1.nii.gz"),
                },
                {
                    "image": os.path.join(tempdir, "test_image2.nii.gz"),
                    "label": os.path.join(tempdir, "test_label2.nii.gz"),
                    "extra": os.path.join(tempdir, "test_extra2.nii.gz"),
                },
            ]

            cache_dir = os.path.join(os.path.join(tempdir, "cache"), "data")
            dataset_precached = LMDBDataset(data=test_data, transform=transform, cache_dir=cache_dir, **kwargs)
            data1_precached = dataset_precached[0]
            data2_precached = dataset_precached[1]

            dataset_postcached = LMDBDataset(data=test_data, transform=transform, cache_dir=cache_dir, **kwargs)
            data1_postcached = dataset_postcached[0]
            data2_postcached = dataset_postcached[1]

        if transform is None:
            self.assertEqual(data1_precached["image"], os.path.join(tempdir, "test_image1.nii.gz"))
            self.assertEqual(data2_precached["label"], os.path.join(tempdir, "test_label2.nii.gz"))
            self.assertEqual(data1_postcached["image"], os.path.join(tempdir, "test_image1.nii.gz"))
            self.assertEqual(data2_postcached["extra"], os.path.join(tempdir, "test_extra2.nii.gz"))
        else:
            self.assertTupleEqual(data1_precached["image"].shape, expected_shape)
            self.assertTupleEqual(data1_precached["label"].shape, expected_shape)
            self.assertTupleEqual(data1_precached["extra"].shape, expected_shape)
            self.assertTupleEqual(data2_precached["image"].shape, expected_shape)
            self.assertTupleEqual(data2_precached["label"].shape, expected_shape)
            self.assertTupleEqual(data2_precached["extra"].shape, expected_shape)

            self.assertTupleEqual(data1_postcached["image"].shape, expected_shape)
            self.assertTupleEqual(data1_postcached["label"].shape, expected_shape)
            self.assertTupleEqual(data1_postcached["extra"].shape, expected_shape)
            self.assertTupleEqual(data2_postcached["image"].shape, expected_shape)
            self.assertTupleEqual(data2_postcached["label"].shape, expected_shape)
            self.assertTupleEqual(data2_postcached["extra"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()

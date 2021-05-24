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

import os
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.data import CacheDataset, DataLoader, PersistentDataset, SmartCacheDataset
from monai.transforms import Compose, LoadImaged, ThreadUnsafe, Transform
from monai.utils import get_torch_version_tuple

TEST_CASE_1 = [Compose([LoadImaged(keys=["image", "label", "extra"])]), (128, 128, 128)]

TEST_CASE_2 = [None, (128, 128, 128)]


TEST_DS = []
for c in (0, 1, 2):
    for l in (0, 1, 2):
        TEST_DS.append([False, c, 0 if sys.platform in ("darwin", "win32") else l])
    if sys.platform not in ("darwin", "win32"):
        # persistent_workers need l > 0
        for l in (1, 2):
            TEST_DS.append([True, c, l])


class TestCacheDataset(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, transform, expected_shape):
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
            dataset = CacheDataset(data=test_data, transform=transform, cache_rate=0.5)
            data1 = dataset[0]
            data2 = dataset[1]
            data3 = dataset[0:-1]
            data4 = dataset[-1]
            self.assertEqual(len(data3), 1)

        if transform is None:
            self.assertEqual(data1["image"], os.path.join(tempdir, "test_image1.nii.gz"))
            self.assertEqual(data2["label"], os.path.join(tempdir, "test_label2.nii.gz"))
            self.assertEqual(data4["image"], os.path.join(tempdir, "test_image2.nii.gz"))
        else:
            self.assertTupleEqual(data1["image"].shape, expected_shape)
            self.assertTupleEqual(data1["label"].shape, expected_shape)
            self.assertTupleEqual(data1["extra"].shape, expected_shape)
            self.assertTupleEqual(data2["image"].shape, expected_shape)
            self.assertTupleEqual(data2["label"].shape, expected_shape)
            self.assertTupleEqual(data2["extra"].shape, expected_shape)
            for d in data3:
                self.assertTupleEqual(d["image"].shape, expected_shape)


class _StatefulTransform(Transform, ThreadUnsafe):
    """
    A transform with an internal state.
    The state is changing at each call.
    """

    def __init__(self):
        self.property = 1

    def __call__(self, data):
        self.property = self.property + 1
        return data * 100 + self.property


class TestCacheThread(unittest.TestCase):
    """
    cache dataset and persistent dataset should behave in the same way when used with different loader settings.
    loader's are tested with two epochs.
    """

    @parameterized.expand(TEST_DS)
    def test_thread_safe(self, persistent_workers, cache_workers, loader_workers):
        expected = [102, 202, 302, 402, 502, 602, 702, 802, 902, 1002]
        _kwg = {"persistent_workers": persistent_workers} if get_torch_version_tuple() > (1, 7) else {}
        data_list = list(range(1, 11))
        dataset = CacheDataset(
            data=data_list,
            transform=_StatefulTransform(),
            cache_rate=1.0,
            num_workers=cache_workers,
            progress=False,
        )
        self.assertListEqual(expected, list(dataset))
        loader = DataLoader(
            CacheDataset(
                data=data_list,
                transform=_StatefulTransform(),
                cache_rate=1.0,
                num_workers=cache_workers,
                progress=False,
            ),
            batch_size=1,
            num_workers=loader_workers,
            **_kwg,
        )
        self.assertListEqual(expected, [y.item() for y in loader])
        self.assertListEqual(expected, [y.item() for y in loader])

        dataset = SmartCacheDataset(
            data=data_list,
            transform=_StatefulTransform(),
            cache_rate=0.7,
            replace_rate=0.5,
            num_replace_workers=cache_workers,
            progress=False,
            shuffle=False,
        )
        self.assertListEqual(expected[:7], list(dataset))
        loader = DataLoader(
            SmartCacheDataset(
                data=data_list,
                transform=_StatefulTransform(),
                cache_rate=0.7,
                replace_rate=0.5,
                num_replace_workers=cache_workers,
                progress=False,
                shuffle=False,
            ),
            batch_size=1,
            num_workers=loader_workers,
            **_kwg,
        )
        self.assertListEqual(expected[:7], [y.item() for y in loader])
        self.assertListEqual(expected[:7], [y.item() for y in loader])

        with tempfile.TemporaryDirectory() as tempdir:
            pdata = PersistentDataset(data=data_list, transform=_StatefulTransform(), cache_dir=tempdir)
            self.assertListEqual(expected, list(pdata))
            loader = DataLoader(
                PersistentDataset(data=data_list, transform=_StatefulTransform(), cache_dir=tempdir),
                batch_size=1,
                num_workers=loader_workers,
                shuffle=False,
                **_kwg,
            )
            self.assertListEqual(expected, [y.item() for y in loader])
            self.assertListEqual(expected, [y.item() for y in loader])


if __name__ == "__main__":
    unittest.main()

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

import sys
import unittest

import numpy as np

from monai.data import DataLoader, GridPatchDataset, PatchIter, PatchIterd
from monai.transforms import RandShiftIntensity, RandShiftIntensityd
from monai.utils import set_determinism


def identity_generator(x):
    # simple transform that returns the input itself
    for idx, item in enumerate(x):
        yield item, idx


class TestGridPatchDataset(unittest.TestCase):
    def setUp(self):
        set_determinism(seed=1234)

    def tearDown(self):
        set_determinism(None)

    def test_shape(self):
        # test Iterable input data
        test_dataset = iter(["vwxyz", "helloworld", "worldfoobar"])
        result = GridPatchDataset(data=test_dataset, patch_iter=identity_generator, with_coordinates=False)
        output = []
        n_workers = 0 if sys.platform == "win32" else 2
        for item in DataLoader(result, batch_size=3, num_workers=n_workers):
            output.append("".join(item))
        if sys.platform == "win32":
            expected = ["ar", "ell", "ldf", "oob", "owo", "rld", "vwx", "wor", "yzh"]
        else:
            expected = ["d", "dfo", "hel", "low", "oba", "orl", "orl", "r", "vwx", "yzw"]
            self.assertEqual(len("".join(expected)), len("".join(list(test_dataset))))
        self.assertEqual(sorted(output), sorted(expected))

    def test_loading_array(self):
        set_determinism(seed=1234)
        # test sequence input data with images
        images = [np.arange(16, dtype=float).reshape(1, 4, 4), np.arange(16, dtype=float).reshape(1, 4, 4)]
        # image level
        patch_intensity = RandShiftIntensity(offsets=1.0, prob=1.0)
        patch_iter = PatchIter(patch_size=(2, 2), start_pos=(0, 0))
        ds = GridPatchDataset(data=images, patch_iter=patch_iter, transform=patch_intensity)
        # use the grid patch dataset
        for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=0):
            np.testing.assert_equal(tuple(item[0].shape), (2, 1, 2, 2))
        np.testing.assert_allclose(
            item[0],
            np.array([[[[1.4965, 2.4965], [5.4965, 6.4965]]], [[[11.3584, 12.3584], [15.3584, 16.3584]]]]),
            rtol=1e-4,
        )
        np.testing.assert_allclose(item[1], np.array([[[0, 1], [0, 2], [2, 4]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5)
        if sys.platform != "win32":
            for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=2):
                np.testing.assert_equal(tuple(item[0].shape), (2, 1, 2, 2))
            np.testing.assert_allclose(
                item[0],
                np.array([[[[1.2548, 2.2548], [5.2548, 6.2548]]], [[[9.1106, 10.1106], [13.1106, 14.1106]]]]),
                rtol=1e-3,
            )
            np.testing.assert_allclose(
                item[1], np.array([[[0, 1], [0, 2], [2, 4]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5
            )

    def test_loading_dict(self):
        set_determinism(seed=1234)
        # test sequence input data with dict
        data = [
            {
                "image": np.arange(16, dtype=float).reshape(1, 4, 4),
                "label": np.arange(16, dtype=float).reshape(1, 4, 4),
                "metadata": "test string",
            },
            {
                "image": np.arange(16, dtype=float).reshape(1, 4, 4),
                "label": np.arange(16, dtype=float).reshape(1, 4, 4),
                "metadata": "test string",
            },
        ]
        # image level
        patch_intensity = RandShiftIntensityd(keys="image", offsets=1.0, prob=1.0)
        patch_iter = PatchIterd(keys=["image", "label"], patch_size=(2, 2), start_pos=(0, 0))
        ds = GridPatchDataset(data=data, patch_iter=patch_iter, transform=patch_intensity, with_coordinates=True)
        # use the grid patch dataset
        for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=0):
            np.testing.assert_equal(item[0]["image"].shape, (2, 1, 2, 2))
            np.testing.assert_equal(item[0]["label"].shape, (2, 1, 2, 2))
            self.assertListEqual(item[0]["metadata"], ["test string", "test string"])
        np.testing.assert_allclose(
            item[0]["image"],
            np.array([[[[1.4965, 2.4965], [5.4965, 6.4965]]], [[[11.3584, 12.3584], [15.3584, 16.3584]]]]),
            rtol=1e-4,
        )
        np.testing.assert_allclose(item[1], np.array([[[0, 1], [0, 2], [2, 4]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5)
        if sys.platform != "win32":
            for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=2):
                np.testing.assert_equal(item[0]["image"].shape, (2, 1, 2, 2))
            np.testing.assert_allclose(
                item[0]["image"],
                np.array([[[[1.2548, 2.2548], [5.2548, 6.2548]]], [[[9.1106, 10.1106], [13.1106, 14.1106]]]]),
                rtol=1e-3,
            )
            np.testing.assert_allclose(
                item[1], np.array([[[0, 1], [0, 2], [2, 4]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5
            )


if __name__ == "__main__":
    unittest.main()

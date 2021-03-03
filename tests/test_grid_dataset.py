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

import sys
import unittest

import numpy as np

from monai.data import DataLoader, GridPatchDataset, PatchIter
from monai.transforms import RandShiftIntensity
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
        test_dataset = ["vwxyz", "helloworld", "worldfoobar"]
        result = GridPatchDataset(dataset=test_dataset, patch_iter=identity_generator, with_coordinates=False)
        output = []
        n_workers = 0 if sys.platform == "win32" else 2
        for item in DataLoader(result, batch_size=3, num_workers=n_workers):
            output.append("".join(item))
        expected = ["vwx", "wor", "yzh", "ldf", "ell", "oob", "owo", "ar", "rld"]
        self.assertEqual(sorted(output), sorted(expected))
        self.assertEqual(len("".join(expected)), len("".join(test_dataset)))

    def test_loading_array(self):
        set_determinism(seed=1234)
        # image dataset
        images = [np.arange(16, dtype=float).reshape(1, 4, 4), np.arange(16, dtype=float).reshape(1, 4, 4)]
        # image level
        patch_intensity = RandShiftIntensity(offsets=1.0, prob=1.0)
        patch_iter = PatchIter(patch_size=(2, 2), start_pos=(0, 0))
        ds = GridPatchDataset(dataset=images, patch_iter=patch_iter, transform=patch_intensity)
        # use the grid patch dataset
        for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=0):
            np.testing.assert_equal(tuple(item[0].shape), (2, 1, 2, 2))
        np.testing.assert_allclose(
            item[0],
            np.array([[[[1.7413, 2.7413], [5.7413, 6.7413]]], [[[9.1419, 10.1419], [13.1419, 14.1419]]]]),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            item[1],
            np.array([[[0, 1], [0, 2], [2, 4]], [[0, 1], [2, 4], [2, 4]]]),
            rtol=1e-5,
        )
        if sys.platform != "win32":
            for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=2):
                np.testing.assert_equal(tuple(item[0].shape), (2, 1, 2, 2))
            np.testing.assert_allclose(
                item[0],
                np.array([[[[2.3944, 3.3944], [6.3944, 7.3944]]], [[[10.6551, 11.6551], [14.6551, 15.6551]]]]),
                rtol=1e-3,
            )
            np.testing.assert_allclose(
                item[1],
                np.array([[[0, 1], [0, 2], [2, 4]], [[0, 1], [2, 4], [2, 4]]]),
                rtol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()

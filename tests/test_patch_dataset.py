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

import unittest

import numpy as np

from monai.data import DataLoader, Dataset, PatchDataset
from monai.transforms import RandShiftIntensity, RandSpatialCropSamples
from monai.utils import set_determinism


def identity(x):
    # simple transform that returns the input itself
    return x


class TestPatchDataset(unittest.TestCase):
    def test_shape(self):
        test_dataset = ["vwxyz", "hello", "world"]
        n_per_image = len(test_dataset[0])

        result = PatchDataset(dataset=test_dataset, patch_func=identity, samples_per_image=n_per_image)

        output = []
        for item in DataLoader(result, batch_size=3, num_workers=5):
            print(item)
            output.append("".join(item))
        expected = ["vwx", "yzh", "ell", "owo", "rld"]
        self.assertEqual(output, expected)

    def test_loading_array(self):
        set_determinism(seed=1234)
        # image dataset
        images = [np.arange(16, dtype=np.float).reshape(1, 4, 4), np.arange(16, dtype=np.float).reshape(1, 4, 4)]
        # image patch sampler
        n_samples = 8
        sampler = RandSpatialCropSamples(roi_size=(3, 3), num_samples=n_samples, random_center=True, random_size=False)

        # image level
        patch_intensity = RandShiftIntensity(offsets=1.0, prob=1.0)
        image_ds = Dataset(images, transform=patch_intensity)
        # patch level
        ds = PatchDataset(dataset=image_ds, patch_func=sampler, samples_per_image=n_samples, transform=patch_intensity)

        np.testing.assert_equal(len(ds), n_samples * len(images))
        # use the patch dataset, length: len(images) x samplers_per_image
        for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=2):
            np.testing.assert_equal(tuple(item.shape), (2, 1, 3, 3))
        np.testing.assert_allclose(
            item[0],
            np.array(
                [[[0.166312, 1.166312, 2.166312], [4.166312, 5.166312, 6.166312], [8.166312, 9.166312, 10.166312]]]
            ),
            rtol=1e-5,
        )
        for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=0):
            np.testing.assert_equal(tuple(item.shape), (2, 1, 3, 3))
        np.testing.assert_allclose(
            item[0],
            np.array(
                [[[1.229009, 2.229009, 3.229009], [5.229009, 6.229009, 7.229009], [9.229009, 10.229009, 11.229009]]]
            ),
            rtol=1e-5,
        )
        set_determinism(seed=None)


if __name__ == "__main__":
    unittest.main()

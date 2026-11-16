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

from monai.apps.pathology import transforms
from monai.apps.pathology.transforms.stain.array import HEDJitter
from monai.apps.pathology.transforms.stain.dictionary import HEDJitterd
from monai.data import CacheDataset
from monai.transforms import Randomizable
from monai.utils import optional_import

_, has_skimage = optional_import("skimage", name="color")


@unittest.skipUnless(has_skimage, "Requires scikit-image")
class TestHEDJitter(unittest.TestCase):
    def setUp(self):
        self.image = np.array(
            [[[90, 50, 100], [200, 140, 180], [120, 80, 160]], [[160, 90, 120], [220, 170, 200], [70, 40, 80]]],
            dtype=np.uint8,
        )

    def test_public_imports(self):
        self.assertIs(getattr(transforms, "HEDJitter", None), HEDJitter)
        for name in ("HEDJitterd", "HEDJitterD", "HEDJitterDict"):
            self.assertIs(getattr(transforms, name, None), HEDJitterd)

    def test_resamples_each_call(self):
        transform = HEDJitter(theta=0.05).set_random_state(seed=123)
        first = transform(self.image)
        second = transform(self.image)
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(first.shape, self.image.shape)
        self.assertEqual(first.dtype, np.uint8)

    def test_seed_reproduces_sequence(self):
        transform = HEDJitter(theta=0.05)
        self.assertIsInstance(transform, Randomizable)
        transform.set_random_state(seed=123)
        expected = [transform(self.image) for _ in range(2)]
        transform.set_random_state(seed=123)
        for image in expected:
            np.testing.assert_array_equal(transform(self.image), image)
        self.assertFalse(np.array_equal(*expected))

    def test_dictionary_shares_factors_and_resamples(self):
        transform = HEDJitterd(keys=("image", "other"), theta=0.05).set_random_state(seed=123)
        original = self.image.copy()
        data = {"image": self.image, "other": self.image.copy(), "label": 1}
        first = transform(data)
        second = transform(data)
        np.testing.assert_array_equal(first["image"], first["other"])
        np.testing.assert_array_equal(second["image"], second["other"])
        self.assertFalse(np.array_equal(first["image"], second["image"]))
        np.testing.assert_array_equal(data["image"], original)
        self.assertEqual(first["label"], 1)

    def test_dictionary_seed_reproduces_sequence(self):
        transform = HEDJitterd(keys="image", theta=0.05)
        self.assertIsInstance(transform, Randomizable)
        state = np.random.RandomState(123)
        transform.set_random_state(state=state)
        data = {"image": self.image}
        expected = [transform(data)["image"] for _ in range(2)]
        transform.set_random_state(seed=123)
        for image in expected:
            np.testing.assert_array_equal(transform(data)["image"], image)
        self.assertFalse(np.array_equal(*expected))

    @parameterized.expand([(False,), (True,)])
    def test_cached_dataset_resamples(self, dictionary):
        transform = HEDJitterd(keys="image", theta=0.05) if dictionary else HEDJitter(theta=0.05)
        data = {"image": self.image} if dictionary else self.image
        dataset = CacheDataset(data=[data], transform=transform, num_workers=1, progress=False)
        dataset.transform.set_random_state(seed=123)
        first, second = dataset[0], dataset[0]
        if dictionary:
            first, second = first["image"], second["image"]
        self.assertFalse(np.array_equal(first, second))

    @parameterized.expand([(0, False), (255, False), (0, True), (255, True)])
    def test_constant_image(self, value, dictionary):
        image = np.full((2, 3, 3), value, dtype=np.uint8)
        transform = HEDJitterd(keys="image") if dictionary else HEDJitter()
        with np.errstate(divide="raise", invalid="raise"):
            result = transform({"image": image})["image"] if dictionary else transform(image)
        np.testing.assert_array_equal(result, image)
        self.assertEqual(result.dtype, np.uint8)

    def test_missing_dictionary_keys(self):
        data = {"image": self.image}
        transform = HEDJitterd(keys=("missing", "image"), theta=0.05, allow_missing_keys=True)
        self.assertEqual(transform(data)["image"].shape, self.image.shape)
        self.assertEqual(transform({}), {})
        with self.assertRaises(KeyError):
            HEDJitterd(keys="missing")(data)


if __name__ == "__main__":
    unittest.main()

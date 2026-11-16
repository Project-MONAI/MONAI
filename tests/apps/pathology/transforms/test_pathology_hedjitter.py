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
    """Exercise stain jitter values, random states, and dictionary integration."""

    def setUp(self):
        """Create a nonconstant uint8 RGB image shared by the test cases."""
        self.image = np.array(
            [[[90, 50, 100], [200, 140, 180], [120, 80, 160]], [[160, 90, 120], [220, 170, 200], [70, 40, 80]]],
            dtype=np.uint8,
        )

    def test_public_imports(self):
        """Expose the array transform and all dictionary aliases publicly."""
        self.assertIs(getattr(transforms, "HEDJitter", None), HEDJitter)
        for name in ("HEDJitterd", "HEDJitterD", "HEDJitterDict"):
            self.assertIs(getattr(transforms, name, None), HEDJitterd)

    def test_resamples_each_call(self):
        """Sample fresh stain factors while retaining the image shape and dtype."""
        transform = HEDJitter(theta=0.05).set_random_state(seed=123)
        first = transform(self.image)
        second = transform(self.image)
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(first.shape, self.image.shape)
        self.assertEqual(first.dtype, np.uint8)

    def test_fixed_stain_factors(self):
        """Apply distinct scale and offset factors in H, E, D order without resampling."""
        transform = HEDJitter(theta=0.05)
        transform.alpha = np.array([[1.1, 0.8, 1.2]])
        transform.betti = np.array([[0.01, -0.005, 0.015]])
        # Independently solve M.T @ stains = -log(RGB / 255) using Ruifrok's
        # stain rows (0.65, 0.70, 0.29), (0.07, 0.99, 0.11), (0.27, 0.57, 0.78).
        # Clip negative concentrations, apply alpha * stains + betti * -log(1e-6),
        # then reconstruct each RGB channel with exp(-sum(stain * coefficient)).
        expected = np.array(
            [[[70, 39, 72], [169, 121, 141], [97, 67, 126]], [[131, 74, 87], [188, 147, 159], [52, 29, 55]]],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(transform(self.image, randomize=False), expected)

    @parameterized.expand([(False,), (True,)])
    def test_zero_jitter_preserves_colors(self, dictionary):
        """Preserve intermediate gray and saturated RGB values when jitter is zero.

        Args:
            dictionary: whether to exercise the dictionary wrapper.
        """
        image = np.array([[[128, 128, 128], [20, 180, 200], [230, 40, 80]]], dtype=np.uint8)
        transform = HEDJitterd(keys="image") if dictionary else HEDJitter()
        result = transform({"image": image})["image"] if dictionary else transform(image)
        np.testing.assert_array_equal(result, image)
        self.assertEqual(result.dtype, np.uint8)
        self.assertFalse(np.shares_memory(result, image))

    def test_seed_reproduces_sequence(self):
        """Reproduce a changing output sequence when resetting the random seed."""
        transform = HEDJitter(theta=0.05)
        self.assertIsInstance(transform, Randomizable)
        transform.set_random_state(seed=123)
        expected = [transform(self.image) for _ in range(2)]
        transform.set_random_state(seed=123)
        for image in expected:
            np.testing.assert_array_equal(transform(self.image), image)
        self.assertFalse(np.array_equal(*expected))

    def test_dictionary_shares_factors_and_resamples(self):
        """Share factors across keys without modifying inputs or unrelated entries."""
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
        """Forward random-state objects and seeds to the wrapped transform."""
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
        """Keep jitter random when a dataset caches its deterministic preprocessing.

        Args:
            dictionary: whether to exercise the dictionary wrapper.
        """
        transform = HEDJitterd(keys="image", theta=0.05) if dictionary else HEDJitter(theta=0.05)
        data = {"image": self.image} if dictionary else self.image
        dataset = CacheDataset(data=[data], transform=transform, num_workers=1, progress=False)
        dataset.transform.set_random_state(seed=123)
        first, second = dataset[0], dataset[0]
        if dictionary:
            first, second = first["image"], second["image"]
        self.assertFalse(np.array_equal(first, second))

    @parameterized.expand([(0, False), (128, False), (255, False), (0, True), (128, True), (255, True)])
    def test_constant_image(self, value, dictionary):
        """Preserve constant RGB images without invalid floating-point operations.

        Args:
            value: intensity to assign to every RGB channel.
            dictionary: whether to exercise the dictionary wrapper.
        """
        image = np.full((2, 3, 3), value, dtype=np.uint8)
        transform = HEDJitterd(keys="image") if dictionary else HEDJitter()
        with np.errstate(divide="raise", invalid="raise"):
            result = transform({"image": image})["image"] if dictionary else transform(image)
        np.testing.assert_array_equal(result, image)
        self.assertEqual(result.dtype, np.uint8)

    def test_missing_dictionary_keys(self):
        """Respect the configured policy for absent dictionary keys."""
        data = {"image": self.image}
        transform = HEDJitterd(keys=("missing", "image"), theta=0.05, allow_missing_keys=True)
        self.assertEqual(transform(data)["image"].shape, self.image.shape)
        self.assertEqual(transform({}), {})
        with self.assertRaises(KeyError):
            HEDJitterd(keys="missing")(data)


if __name__ == "__main__":
    unittest.main()

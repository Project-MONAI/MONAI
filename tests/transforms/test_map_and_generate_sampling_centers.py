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
from copy import deepcopy

import numpy as np
from parameterized import parameterized

from monai.transforms import map_and_generate_sampling_centers
from monai.transforms.utils import (
    _map_classes_to_lazy_indices,
    generate_label_classes_crop_centers,
    map_classes_to_indices,
)
from monai.utils.misc import set_determinism
from tests.test_utils import TEST_NDARRAYS, assert_allclose

TEST_CASE_1 = [
    # test Argmax data
    {
        "label": (np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]])),
        "spatial_size": [2, 2, 2],
        "num_samples": 2,
        "label_spatial_shape": [3, 3, 3],
        "num_classes": 3,
        "image": None,
        "ratios": [0, 1, 2],
        "image_threshold": 0.0,
    },
    tuple,
    2,
    3,
]

TEST_CASE_2 = [
    {
        "label": (
            np.array(
                [
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                ]
            )
        ),
        "spatial_size": [2, 2, 2],
        "num_samples": 1,
        "ratios": None,
        "label_spatial_shape": [3, 3, 3],
        "image": None,
        "image_threshold": 0.0,
    },
    tuple,
    1,
    3,
]


class TestMapAndGenerateSamplingCenters(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_map_and_generate_sampling_centers(self, input_data, expected_type, expected_count, expected_shape):
        results = []
        for p in TEST_NDARRAYS + (None,):
            input_data = deepcopy(input_data)
            if p is not None:
                input_data["label"] = p(input_data["label"])
            set_determinism(0)
            result = map_and_generate_sampling_centers(**input_data)
            self.assertIsInstance(result, expected_type)
            self.assertEqual(len(result), expected_count)
            self.assertEqual(len(result[0]), expected_shape)
            # check for consistency between numpy, torch and torch.cuda
            results.append(result)
            if len(results) > 1:
                for x, y in zip(result[0], result[-1]):
                    assert_allclose(x, y, type_test=False)


def _labels_for_lazy_indices_test():
    rng = np.random.default_rng(0)
    shape = (1, 6, 7, 8)
    float_label = rng.integers(-1, 6, size=shape).astype(np.float32)
    float_label[..., ::5] += 0.5  # not a class
    float_label[..., ::7] = np.nan
    one_hot = np.stack([rng.integers(0, 3, size=shape[1:]) == c for c in range(4)]).astype(np.float32)
    return [
        [rng.integers(0, 4, size=shape).astype(np.uint8), 4, None],
        [rng.integers(0, 7, size=shape).astype(np.int64), 5, 3],  # values >= num_classes are not a class
        [rng.integers(-3, 4, size=shape).astype(np.int16), 4, None],
        [float_label, 5, None],
        [np.full(shape, 300, dtype=np.float32), 3, None],  # no voxel of any class
        [one_hot, None, 2],
    ]


class TestLazyClassIndices(unittest.TestCase):
    @parameterized.expand(_labels_for_lazy_indices_test())
    def test_same_as_map_classes_to_indices(self, label, num_classes, max_samples_per_class):
        image = np.random.default_rng(1).random((1, *label.shape[1:])).astype(np.float32)
        for p in TEST_NDARRAYS:
            for img in (None, p(image)):
                expected = map_classes_to_indices(p(label), num_classes, img, 0.3, max_samples_per_class)
                lazy = _map_classes_to_lazy_indices(p(label), num_classes, img, 0.3, max_samples_per_class)
                self.assertEqual([len(i) for i in lazy], [len(i) for i in expected])
                for e, lz in zip(expected, lazy):
                    for k in range(len(e)):
                        assert_allclose(lz[k], e[k], type_test=True)

    @parameterized.expand(_labels_for_lazy_indices_test())
    def test_same_centers(self, label, num_classes, max_samples_per_class):
        indices = map_classes_to_indices(label, num_classes, None, 0.0, max_samples_per_class)
        if not any(len(i) for i in indices):
            return
        for seed in range(3):
            expected = generate_label_classes_crop_centers(
                (2, 2, 2), 4, label.shape[1:], indices, rand_state=np.random.RandomState(seed), warn=False
            )
            result = map_and_generate_sampling_centers(
                label,
                (2, 2, 2),
                4,
                num_classes=num_classes,
                max_samples_per_class=max_samples_per_class,
                rand_state=np.random.RandomState(seed),
                warn=False,
            )
            self.assertEqual(result, expected)

    def test_only_sampled_classes_are_computed(self):
        label = np.random.default_rng(0).integers(0, 10, size=(1, 8, 8, 8))
        lazy = _map_classes_to_lazy_indices(label, 10)
        generate_label_classes_crop_centers((2, 2, 2), 1, label.shape[1:], lazy, rand_state=np.random.RandomState(0))
        self.assertEqual(sum(i._indices is not None for i in lazy), 1)


if __name__ == "__main__":
    unittest.main()

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

import unittest
from copy import deepcopy

import numpy as np
import torch
from parameterized import parameterized

from monai.data.synthetic import create_test_image_2d, create_test_image_3d
from monai.transforms import RandKSpaceSpikeNoised
from monai.utils.misc import set_determinism
from tests.utils import SkipIfBeforePyTorchVersion, SkipIfNoModule

TEST_CASES = []
for shape in ((128, 64), (64, 48, 80)):
    for as_tensor_output in (True, False):
        for as_tensor_input in (True, False):
            TEST_CASES.append((shape, as_tensor_output, as_tensor_input))

KEYS = ["image", "label"]


@SkipIfBeforePyTorchVersion((1, 8))
@SkipIfNoModule("torch.fft")
class TestKSpaceSpikeNoised(unittest.TestCase):
    def setUp(self):
        set_determinism(0)
        super().setUp()

    def tearDown(self):
        set_determinism(None)

    @staticmethod
    def get_data(im_shape, as_tensor_input):
        create_test_image = create_test_image_2d if len(im_shape) == 2 else create_test_image_3d
        ims = create_test_image(*im_shape, rad_max=20, noise_max=0.0, num_seg_classes=5)
        ims = [im[None] for im in ims]
        ims = [torch.Tensor(im) for im in ims] if as_tensor_input else ims
        return {k: v for k, v in zip(KEYS, ims)}

    @parameterized.expand(TEST_CASES)
    def test_same_result(self, im_shape, as_tensor_output, as_tensor_input):

        data = self.get_data(im_shape, as_tensor_input)

        intensity_ranges = {"image": (13, 15), "label": (13, 15)}
        t = RandKSpaceSpikeNoised(
            KEYS,
            global_prob=1.0,
            prob=1.0,
            intensity_ranges=intensity_ranges,
            channel_wise=True,
            as_tensor_output=as_tensor_output,
        )
        t.set_rand_state(42)
        out1 = t(deepcopy(data))

        t.set_rand_state(42)
        out2 = t(deepcopy(data))

        for k in KEYS:
            np.testing.assert_allclose(out1[k], out2[k], atol=1e-10)
            self.assertIsInstance(out1[k], torch.Tensor if as_tensor_output else np.ndarray)

    @parameterized.expand(TEST_CASES)
    def test_0_prob(self, im_shape, as_tensor_output, as_tensor_input):
        data = self.get_data(im_shape, as_tensor_input)
        intensity_ranges = {"image": (13, 15), "label": (13, 15)}
        t1 = RandKSpaceSpikeNoised(
            KEYS,
            global_prob=0.0,
            prob=1.0,
            intensity_ranges=intensity_ranges,
            channel_wise=True,
            as_tensor_output=as_tensor_output,
        )

        t2 = RandKSpaceSpikeNoised(
            KEYS,
            global_prob=0.0,
            prob=1.0,
            intensity_ranges=intensity_ranges,
            channel_wise=True,
            as_tensor_output=as_tensor_output,
        )
        out1 = t1(data)
        out2 = t2(data)

        for k in KEYS:
            np.testing.assert_allclose(data[k], out1[k])
            np.testing.assert_allclose(data[k], out2[k])

    @parameterized.expand(TEST_CASES)
    def test_intensity(self, im_shape, as_tensor_output, as_tensor_input):

        data = self.get_data(im_shape, as_tensor_input)
        intensity_ranges = {"image": (13, 13.1), "label": (13, 13.1)}
        t = RandKSpaceSpikeNoised(
            KEYS,
            global_prob=1.0,
            prob=1.0,
            intensity_ranges=intensity_ranges,
            channel_wise=True,
            as_tensor_output=True,
        )

        _ = t(data)
        self.assertGreaterEqual(t.transforms["image"].sampled_k_intensity[0], 13)
        self.assertLessEqual(t.transforms["image"].sampled_k_intensity[0], 13.1)
        self.assertGreaterEqual(t.transforms["label"].sampled_k_intensity[0], 13)
        self.assertLessEqual(t.transforms["label"].sampled_k_intensity[0], 13.1)

    @parameterized.expand(TEST_CASES)
    def test_same_transformation(self, im_shape, _, as_tensor_input):
        data = self.get_data(im_shape, as_tensor_input)
        # use same image for both dictionary entries to check same trans is applied to them
        data = {KEYS[0]: deepcopy(data[KEYS[0]]), KEYS[1]: deepcopy(data[KEYS[0]])}

        intensity_ranges = {"image": (13, 15), "label": (13, 15)}
        # use common_sampling = True to ask for the same transformation
        t = RandKSpaceSpikeNoised(
            KEYS,
            global_prob=1.0,
            prob=1.0,
            intensity_ranges=intensity_ranges,
            channel_wise=True,
            common_sampling=True,
            as_tensor_output=True,
        )

        out = t(deepcopy(data))

        np.testing.assert_allclose(out[KEYS[0]], out[KEYS[1]])


if __name__ == "__main__":
    unittest.main()

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
from tests.utils import TEST_NDARRAYS

TESTS = []
for shape in ((128, 64), (64, 48, 80)):
    for p in TEST_NDARRAYS:
        TESTS.append((shape, p))

KEYS = ["image", "label"]


class TestKSpaceSpikeNoised(unittest.TestCase):
    def setUp(self):
        set_determinism(0)
        super().setUp()

    def tearDown(self):
        set_determinism(None)

    @staticmethod
    def get_data(im_shape, im_type):
        create_test_image = create_test_image_2d if len(im_shape) == 2 else create_test_image_3d
        ims = create_test_image(*im_shape, rad_max=20, noise_max=0.0, num_seg_classes=5)
        ims = [im_type(im[None]) for im in ims]
        return {k: v for k, v in zip(KEYS, ims)}

    @parameterized.expand(TESTS)
    def test_same_result(self, im_shape, im_type):

        data = self.get_data(im_shape, im_type)

        intensity_range = (13, 15)
        t = RandKSpaceSpikeNoised(
            KEYS,
            global_prob=1.0,
            prob=1.0,
            img_intensity_range=intensity_range,
            label_intensity_range=intensity_range,
            channel_wise=True,
        )
        t.set_rand_state(42)
        out1 = t(deepcopy(data))

        t.set_rand_state(42)
        out2 = t(deepcopy(data))

        for k in KEYS:
            self.assertEqual(type(out1[k]), type(data[k]))
            if isinstance(out1[k], torch.Tensor):
                self.assertEqual(out1[k].device, data[k].device)
                out1[k] = out1[k].cpu()
                out2[k] = out2[k].cpu()
            np.testing.assert_allclose(out1[k], out2[k], atol=1e-10)

    @parameterized.expand(TESTS)
    def test_0_prob(self, im_shape, im_type):
        data = self.get_data(im_shape, im_type)
        intensity_range = (13, 15)
        t1 = RandKSpaceSpikeNoised(
            KEYS,
            global_prob=0.0,
            prob=1.0,
            img_intensity_range=intensity_range,
            label_intensity_range=intensity_range,
            channel_wise=True,
        )

        t2 = RandKSpaceSpikeNoised(
            KEYS,
            global_prob=0.0,
            prob=1.0,
            img_intensity_range=intensity_range,
            label_intensity_range=intensity_range,
            channel_wise=True,
        )
        out1 = t1(data)
        out2 = t2(data)

        for k in KEYS:
            self.assertEqual(type(out1[k]), type(data[k]))
            if isinstance(out1[k], torch.Tensor):
                self.assertEqual(out1[k].device, data[k].device)
                out1[k] = out1[k].cpu()
                out2[k] = out2[k].cpu()
                data[k] = data[k].cpu()

            np.testing.assert_allclose(data[k], out1[k])
            np.testing.assert_allclose(data[k], out2[k])

    @parameterized.expand(TESTS)
    def test_intensity(self, im_shape, im_type):

        data = self.get_data(im_shape, im_type)
        image_range = (15, 15.1)
        label_range = (14, 14.1)
        t = RandKSpaceSpikeNoised(
            KEYS,
            global_prob=1.0,
            prob=1.0,
            img_intensity_range=image_range,
            label_intensity_range=label_range,
            channel_wise=True,
        )

        _ = t(data)
        self.assertGreaterEqual(t.t_img.sampled_k_intensity[0], 15)
        self.assertLessEqual(t.t_img.sampled_k_intensity[0], 15.1)
        self.assertGreaterEqual(t.t_label.sampled_k_intensity[0], 14)
        self.assertLessEqual(t.t_label.sampled_k_intensity[0], 14.1)

    @parameterized.expand(TESTS)
    def test_same_transformation(self, im_shape, im_type):
        data = self.get_data(im_shape, im_type)
        # use same image for both dictionary entries to check same trans is applied to them
        data = {KEYS[0]: deepcopy(data[KEYS[0]]), KEYS[1]: deepcopy(data[KEYS[0]])}

        image_range = label_range = (15, 15.1)
        # use common_sampling = True to ask for the same transformation
        t = RandKSpaceSpikeNoised(
            KEYS,
            global_prob=1.0,
            prob=1.0,
            img_intensity_range=image_range,
            label_intensity_range=label_range,
            channel_wise=True,
            common_sampling=True,
        )

        out = t(deepcopy(data))

        for k in KEYS:
            self.assertEqual(type(out[k]), type(data[k]))
            if isinstance(out[k], torch.Tensor):
                self.assertEqual(out[k].device, data[k].device)
                out[k] = out[k].cpu()

        np.testing.assert_allclose(out[KEYS[0]], out[KEYS[1]])


if __name__ == "__main__":
    unittest.main()

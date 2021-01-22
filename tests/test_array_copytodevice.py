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

import numpy as np
from parameterized import parameterized
from tests.utils import skip_if_no_cuda

from monai.data import ArrayDataset
from monai.transforms import Compose, CopyToDevice, ToTensor

DEVICE="cuda:0"

TEST_CASE_0 = [
    Compose([ToTensor(), CopyToDevice(device=DEVICE)]),
    Compose([ToTensor()]),
    DEVICE,
    "cpu",
]

@skip_if_no_cuda
class TestArrayCopyToDevice(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0])
    def test_array_copy_to_device(self, img_transform, label_transform, img_device, label_device):
        numel = 2
        test_imgs = [np.zeros((3,3,3)) for _ in range(numel)]
        test_segs = [np.zeros((3,3,3)) for _ in range(numel)]

        test_labels = [1, 1]
        dataset = ArrayDataset(test_imgs, img_transform, test_segs, label_transform, test_labels, None)
        self.assertEqual(len(dataset), 2)
        for data in dataset:
            im, seg = data[0], data[1]
            self.assertTrue(str(im.device) == img_device)
            self.assertTrue(str(seg.device) == label_device)



if __name__ == "__main__":
    unittest.main()

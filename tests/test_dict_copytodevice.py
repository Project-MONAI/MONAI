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

from monai.data import Dataset
from monai.transforms import Compose, CopyToDeviced, ToTensord
from tests.utils import skip_if_no_cuda

DEVICE = "cuda:0"

TEST_CASE_0 = [
    Compose([ToTensord(keys=["image", "label", "other"]), CopyToDeviced(keys=["image", "label"], device=DEVICE)]),
    DEVICE,
    "cpu",
]


@skip_if_no_cuda
class TestDictCopyToDevice(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0])
    def test_dict_copy_to_device(self, transform, modified_device, unmodified_device):

        numel = 2
        test_data = [
            {
                "image": np.zeros((3, 3, 3)),
                "label": np.zeros((3, 3, 3)),
                "other": np.zeros((3, 3, 3)),
            }
            for _ in range(numel)
        ]

        dataset = Dataset(data=test_data, transform=transform)
        self.assertEqual(len(dataset), 2)
        for data in dataset:
            self.assertTrue(str(data["image"].device) == modified_device)
            self.assertTrue(str(data["label"].device) == modified_device)
            self.assertTrue(str(data["other"].device) == unmodified_device)


if __name__ == "__main__":
    unittest.main()

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

from monai.data import DataLoader
from monai.data import CacheDataset, create_test_image_2d, create_test_image_3d
from monai.transforms import AddChanneld, Compose, LoadImaged, ToTensord
from monai.data.utils import decollate_batch
from monai.utils import set_determinism
from tests.utils import make_nifti_image

from parameterized import parameterized


set_determinism(seed=0)

IM_2D_FNAME = make_nifti_image(create_test_image_2d(100, 101)[0])
IM_3D_FNAME = make_nifti_image(create_test_image_3d(100, 101, 107)[0])

TRANSFORMS = Compose([LoadImaged("image"), AddChanneld("image"), ToTensord("image")])
DATA_2D = {"image": IM_2D_FNAME}
DATA_3D = {"image": IM_3D_FNAME}

TESTS = []
TESTS.append((
    "2D",
    [DATA_2D for _ in range(5)],
    TRANSFORMS,
))
TESTS.append((
    "3D",
    [DATA_3D for _ in range(9)],
    TRANSFORMS,
))

class TestDeCollate(unittest.TestCase):
    def check_dictionaries_match(self, d1, d2):
        self.assertEqual(d1.keys(), d2.keys())
        for v1, v2 in zip(d1.values(), d2.values()):
            if isinstance(v1, dict):
                self.check_dictionaries_match(v1, v2)
            else:
                np.testing.assert_array_equal(v1, v2)

    @parameterized.expand(TESTS)
    def test_decollation(self, _, data, transforms, batch_size=2, num_workers=0):
        dataset = CacheDataset(data, transforms, progress=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        for b, batch_data in enumerate(loader):
            decollated = decollate_batch(batch_data)

            for i, d in enumerate(decollated):
                self.check_dictionaries_match(d, dataset[b * batch_size + i])


if __name__ == "__main__":
    unittest.main()

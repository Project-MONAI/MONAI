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
import torch
from parameterized import parameterized

from monai.data import CacheDataset, DataLoader, create_test_image_2d
from monai.data.utils import decollate_batch
from monai.transforms import AddChanneld, Compose, LoadImaged, RandFlipd, SpatialPadd, ToTensord
from monai.transforms.post.dictionary import Decollated
from monai.utils import optional_import, set_determinism
from tests.utils import make_nifti_image

_, has_nib = optional_import("nibabel")

IM_2D = create_test_image_2d(100, 101)[0]
DATA_2D = {"image": make_nifti_image(IM_2D) if has_nib else IM_2D}

TESTS = []
TESTS.append(
    (
        "2D",
        [DATA_2D for _ in range(6)],
    )
)


class TestDeCollate(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(seed=0)

    def tearDown(self) -> None:
        set_determinism(None)

    def check_match(self, in1, in2):
        if isinstance(in1, dict):
            self.assertTrue(isinstance(in2, dict))
            self.check_match(list(in1.keys()), list(in2.keys()))
            self.check_match(list(in1.values()), list(in2.values()))
        elif any(isinstance(in1, i) for i in [list, tuple]):
            for l1, l2 in zip(in1, in2):
                self.check_match(l1, l2)
        elif any(isinstance(in1, i) for i in [str, int]):
            self.assertEqual(in1, in2)
        elif any(isinstance(in1, i) for i in [torch.Tensor, np.ndarray]):
            np.testing.assert_array_equal(in1, in2)
        else:
            raise RuntimeError(f"Not sure how to compare types. type(in1): {type(in1)}, type(in2): {type(in2)}")

    @parameterized.expand(TESTS)
    def test_decollation(self, _, data, batch_size=2, num_workers=2):
        transforms = Compose(
            [
                AddChanneld("image"),
                SpatialPadd("image", 150),
                RandFlipd("image", prob=1.0, spatial_axis=1),
                ToTensord("image"),
            ]
        )
        # If nibabel present, read from disk
        if has_nib:
            transforms = Compose([LoadImaged("image"), transforms])

        dataset = CacheDataset(data, transforms, progress=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for b, batch_data in enumerate(loader):
            decollated_1 = decollate_batch(batch_data)
            decollated_2 = Decollated()(batch_data)

            for z, decollated in enumerate([decollated_1, decollated_2]):
                for i, d in enumerate(decollated):
                    try:
                        self.check_match(dataset[b * batch_size + i], d)
                    except RuntimeError:
                        print(f"problem with b={b}, i={i}, decollated_{z+1}")
                        print("d")
                        print(d)
                        print("dataset[b * batch_size + i]")
                        print(dataset[b * batch_size + i])
                        raise


if __name__ == "__main__":
    unittest.main()

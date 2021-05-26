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
from typing import List, Sequence, Tuple

from parameterized import parameterized

from monai.data import create_test_image_2d
from monai.transforms import AddChanneld, Compose, LoadImaged, RandAxisFlipd, RandFlipd
from monai.utils import set_determinism
from tests.utils import make_nifti_image

KEYS = ["image", "label"]

TESTS: List[Tuple] = []


class ErrorRandAxisFlipd(RandAxisFlipd):
    def inverse(self, _):
        raise RuntimeError


# remove the ErrorRandAxisFlipd transform. Since its inverse
# raises an exception, we'll know if this wasn't successful
TESTS.append(
    (
        Compose(
            [
                LoadImaged(KEYS),
                AddChanneld(KEYS),
                RandAxisFlipd("image"),
                ErrorRandAxisFlipd("label"),
                ErrorRandAxisFlipd(KEYS),
                RandFlipd(KEYS),
            ]
        ),
        "ErrorRandAxisFlipd",
        False,
    )
)

# Nothing is removed, so exception is expected
TESTS.append(
    (
        Compose([LoadImaged(KEYS), AddChanneld(KEYS), ErrorRandAxisFlipd("label")]),
        "",
        True,
    )
)


class TestInverseSubset(unittest.TestCase):
    def setUp(self):
        set_determinism(seed=0)

        im_fnames = [make_nifti_image(i) for i in create_test_image_2d(101, 100)]
        self.data = {k: v for k, v in zip(KEYS, im_fnames)}

    def tearDown(self):
        set_determinism(seed=None)

    @parameterized.expand(TESTS)
    def test_inverse_subset(
        self,
        transforms: Compose,
        to_skip: Sequence[str],
        expected_exception: bool,
    ) -> None:
        d = transforms(self.data)
        if not expected_exception:
            transforms.inverse_with_omissions(d, to_skip)
        else:
            with self.assertRaises(RuntimeError):
                transforms.inverse_with_omissions(d, to_skip)


if __name__ == "__main__":
    unittest.main()

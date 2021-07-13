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

import sys
import unittest

import numpy as np
import torch

from monai.data import CacheDataset, DataLoader, create_test_image_3d, decollate_batch
from monai.transforms import (
    AddChanneld,
    CastToTyped,
    Compose,
    CopyItemsd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandAxisFlipd,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    Spacingd,
)
from monai.utils.misc import set_determinism
from tests.utils import make_nifti_image

KEYS = ["image", "label"]


class TestInvertd(unittest.TestCase):
    def test_invert(self):
        set_determinism(seed=0)
        im_fname, seg_fname = [make_nifti_image(i) for i in create_test_image_3d(101, 100, 107, noise_max=100)]
        transform = Compose(
            [
                LoadImaged(KEYS),
                AddChanneld(KEYS),
                Orientationd(KEYS, "RPS"),
                Spacingd(KEYS, pixdim=(1.2, 1.01, 0.9), mode=["bilinear", "nearest"], dtype=np.float32),
                ScaleIntensityd("image", minv=1, maxv=10),
                RandFlipd(KEYS, prob=0.5, spatial_axis=[1, 2]),
                RandAxisFlipd(KEYS, prob=0.5),
                RandRotate90d(KEYS, spatial_axes=(1, 2)),
                RandZoomd(KEYS, prob=0.5, min_zoom=0.5, max_zoom=1.1, keep_size=True),
                RandRotated(KEYS, prob=0.5, range_x=np.pi, mode="bilinear", align_corners=True),
                RandAffined(KEYS, prob=0.5, rotate_range=np.pi, mode="nearest"),
                ResizeWithPadOrCropd(KEYS, 100),
                # test EnsureTensor for complicated dict data and invert it
                CopyItemsd("image_meta_dict", times=1, names="test_dict"),
                # test to support Tensor, Numpy array and dictionary when inverting
                EnsureTyped(keys=["image", "test_dict"]),
                CastToTyped(KEYS, dtype=[torch.uint8, np.uint8]),
                CopyItemsd("label", times=1, names="label_inverted"),
            ]
        )
        data = [{"image": im_fname, "label": seg_fname} for _ in range(12)]

        # num workers = 0 for mac or gpu transforms
        num_workers = 0 if sys.platform != "linux" or torch.cuda.is_available() else 2

        dataset = CacheDataset(data, transform=transform, progress=False)
        loader = DataLoader(dataset, num_workers=num_workers, batch_size=5)
        inverter = Invertd(
            # `image` was not copied, invert the original value directly
            keys=["image", "label_inverted", "test_dict"],
            transform=transform,
            orig_keys=["label", "label", "test_dict"],
            meta_keys=["image_meta_dict", "label_inverted_meta_dict", None],
            orig_meta_keys=["label_meta_dict", "label_meta_dict", None],
            nearest_interp=True,
            to_tensor=[True, False, False],
            device="cpu",
            num_workers=num_workers,
        )

        # execute 1 epoch
        for d in loader:
            d = decollate_batch(d)
            for item in d:
                item = inverter(item)
                # this unit test only covers basic function, test_handler_transform_inverter covers more
                self.assertTupleEqual(item["label"].shape[1:], (100, 100, 100))
                # check the nearest inerpolation mode
                i = item["image"]
                torch.testing.assert_allclose(i.to(torch.uint8).to(torch.float), i.to(torch.float))
                self.assertTupleEqual(i.shape[1:], (100, 101, 107))
                i = item["label_inverted"]
                np.testing.assert_allclose(i.astype(np.uint8).astype(np.float32), i.astype(np.float32))
                self.assertTupleEqual(i.shape[1:], (100, 101, 107))
                # test inverted test_dict
                self.assertTrue(isinstance(item["test_dict"]["affine"], np.ndarray))
                self.assertTrue(isinstance(item["test_dict"]["filename_or_obj"], str))

        set_determinism(seed=None)


if __name__ == "__main__":
    unittest.main()

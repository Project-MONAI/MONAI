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

import sys
import unittest

import numpy as np
import torch

from monai.data import DataLoader, Dataset, MetaTensor, create_test_image_2d, create_test_image_3d, decollate_batch
from monai.transforms import (
    CastToTyped,
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
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
from monai.transforms.utility.dictionary import Lambdad
from monai.utils import set_determinism
from tests.test_utils import assert_allclose, make_nifti_image

KEYS = ["image", "label"]


class TestInvertd(unittest.TestCase):
    def test_invert(self):
        set_determinism(seed=0)
        im_fname, seg_fname = (make_nifti_image(i) for i in create_test_image_3d(101, 100, 107, noise_max=100))
        transform = Compose(
            [
                LoadImaged(KEYS, image_only=True),
                EnsureChannelFirstd(KEYS),
                Orientationd(KEYS, "RPS"),
                Spacingd(KEYS, pixdim=(1.2, 1.01, 0.9), mode=["bilinear", "nearest"], dtype=np.float32),
                ScaleIntensityd("image", minv=1, maxv=10),
                RandFlipd(KEYS, prob=0.5, spatial_axis=[1, 2]),
                RandAxisFlipd(KEYS, prob=0.5),
                RandRotate90d(KEYS, prob=0, spatial_axes=(1, 2)),
                RandZoomd(KEYS, prob=0.5, min_zoom=0.5, max_zoom=1.1, keep_size=True),
                RandRotated(KEYS, prob=0.5, range_x=np.pi, mode="bilinear", align_corners=True, dtype=np.float64),
                RandAffined(KEYS, prob=0.5, rotate_range=np.pi, mode=["nearest", 0]),
                ResizeWithPadOrCropd(KEYS, 100),
                CastToTyped(KEYS, dtype=[torch.uint8, np.uint8]),
                CopyItemsd("label", times=2, names=["label_inverted", "label_inverted1"]),
                CopyItemsd("image", times=2, names=["image_inverted", "image_inverted1"]),
            ]
        )
        data = [{"image": im_fname, "label": seg_fname} for _ in range(12)]

        # num workers = 0 for mac or gpu transforms
        num_workers = 0 if sys.platform != "linux" or torch.cuda.is_available() else 2

        dataset = Dataset(data, transform=transform)
        transform.inverse(dataset[0])
        loader = DataLoader(dataset, num_workers=num_workers, batch_size=1)
        inverter = Invertd(
            # `image` was not copied, invert the original value directly
            keys=["image_inverted", "label_inverted"],
            transform=transform,
            orig_keys=["label", "label"],
            nearest_interp=True,
            device=None,
            post_func=torch.as_tensor,
        )

        inverter_1 = Invertd(
            # `image` was not copied, invert the original value directly
            keys=["image_inverted1", "label_inverted1"],
            transform=transform,
            orig_keys=["image", "image"],
            nearest_interp=[True, False],
            device="cpu",
        )

        expected_keys = ["image", "image_inverted", "image_inverted1", "label", "label_inverted", "label_inverted1"]
        # execute 1 epoch
        for d in loader:
            d = decollate_batch(d)
            for item in d:
                item = inverter(item)
                item = inverter_1(item)

                self.assertListEqual(sorted(item), expected_keys)
                self.assertTupleEqual(item["image"].shape[1:], (100, 100, 100))
                self.assertTupleEqual(item["label"].shape[1:], (100, 100, 100))
                # check the nearest interpolation mode
                i = item["image_inverted"]
                assert_allclose(i.to(torch.uint8).to(torch.float), i.to(torch.float))
                self.assertTupleEqual(i.shape[1:], (101, 100, 107))
                i = item["label_inverted"]
                assert_allclose(i.to(torch.uint8).to(torch.float), i.to(torch.float))
                self.assertTupleEqual(i.shape[1:], (101, 100, 107))

                # check the case that different items use different interpolation mode to invert transforms
                j = item["image_inverted1"]
                # if the interpolation mode is nearest, accumulated diff should be smaller than 1
                self.assertLess(torch.sum(j.to(torch.float) - j.to(torch.uint8).to(torch.float)).item(), 1.0)
                self.assertTupleEqual(j.shape, (1, 101, 100, 107))

                k = item["label_inverted1"]
                # if the interpolation mode is not nearest, accumulated diff should be greater than 10000
                self.assertGreater(torch.sum(k.to(torch.float) - k.to(torch.uint8).to(torch.float)).item(), 10000.0)
                self.assertTupleEqual(k.shape, (1, 101, 100, 107))

        # check labels match
        reverted = item["label_inverted"].detach().cpu().numpy().astype(np.int32)
        original = LoadImaged(KEYS, image_only=True)(data[-1])["label"]
        n_good = np.sum(np.isclose(reverted, original, atol=1e-3))
        reverted_name = item["label_inverted"].meta["filename_or_obj"]
        original_name = data[-1]["label"]
        self.assertEqual(reverted_name, original_name)
        print("invert diff", reverted.size - n_good)
        # 25300: 2 workers (cpu, non-macos)
        # 1812: 0 workers (gpu or macos)
        # 1821: windows torch 1.10.0
        self.assertLess((reverted.size - n_good), 40000, f"diff.  {reverted.size - n_good}")

        set_determinism(seed=None)

    def test_invertd_with_postprocessing_transforms(self):
        """Test that Invertd ignores postprocessing transforms using automatic group tracking.

        This is a regression test for the issue where Invertd would fail when
        postprocessing contains invertible transforms before Invertd is called.
        The fix uses automatic group tracking where Compose assigns its ID to child transforms.
        """
        img, _ = create_test_image_2d(60, 60, 2, 10, num_seg_classes=2)
        img = MetaTensor(img, meta={"original_channel_dim": float("nan"), "pixdim": [1.0, 1.0, 1.0]})
        key = "image"

        # Preprocessing pipeline
        preprocessing = Compose([EnsureChannelFirstd(key), Spacingd(key, pixdim=[2.0, 2.0])])

        # Postprocessing with Lambdad before Invertd
        # Previously this would raise RuntimeError about transform ID mismatch
        postprocessing = Compose(
            [
                Lambdad(key, func=lambda x: x),  # Should be ignored during inversion
                Invertd(key, transform=preprocessing, orig_keys=key),
            ]
        )

        # Apply transforms
        item = {key: img}
        pre = preprocessing(item)

        # This should NOT raise an error (was failing before the fix)
        try:
            post = postprocessing(pre)
            # If we get here, the bug is fixed
            self.assertIsNotNone(post)
            self.assertIn(key, post)
        except RuntimeError as e:
            if "getting the most recently applied invertible transform" in str(e):
                self.fail(f"Invertd still has the postprocessing transform bug: {e}")

    def test_invertd_multiple_pipelines(self):
        """Test that Invertd correctly handles multiple independent preprocessing pipelines."""
        img1, _ = create_test_image_2d(60, 60, 2, 10, num_seg_classes=2)
        img1 = MetaTensor(img1, meta={"original_channel_dim": float("nan"), "pixdim": [1.0, 1.0, 1.0]})
        img2, _ = create_test_image_2d(60, 60, 2, 10, num_seg_classes=2)
        img2 = MetaTensor(img2, meta={"original_channel_dim": float("nan"), "pixdim": [1.0, 1.0, 1.0]})

        # Two different preprocessing pipelines
        preprocessing1 = Compose([EnsureChannelFirstd("image1"), Spacingd("image1", pixdim=[2.0, 2.0])])

        preprocessing2 = Compose([EnsureChannelFirstd("image2"), Spacingd("image2", pixdim=[1.5, 1.5])])

        # Postprocessing that inverts both
        postprocessing = Compose(
            [
                Lambdad(["image1", "image2"], func=lambda x: x),
                Invertd("image1", transform=preprocessing1, orig_keys="image1"),
                Invertd("image2", transform=preprocessing2, orig_keys="image2"),
            ]
        )

        # Apply transforms
        item = {"image1": img1, "image2": img2}
        pre1 = preprocessing1(item)
        pre2 = preprocessing2(pre1)

        # Should not raise error - each Invertd should only invert its own pipeline
        post = postprocessing(pre2)
        self.assertIn("image1", post)
        self.assertIn("image2", post)

    def test_invertd_multiple_postprocessing_transforms(self):
        """Test Invertd with multiple invertible transforms in postprocessing before Invertd."""
        img, _ = create_test_image_2d(60, 60, 2, 10, num_seg_classes=2)
        img = MetaTensor(img, meta={"original_channel_dim": float("nan"), "pixdim": [1.0, 1.0, 1.0]})
        key = "image"

        preprocessing = Compose([EnsureChannelFirstd(key), Spacingd(key, pixdim=[2.0, 2.0])])

        # Multiple transforms in postprocessing before Invertd
        postprocessing = Compose(
            [
                Lambdad(key, func=lambda x: x * 2),
                Lambdad(key, func=lambda x: x + 1),
                Lambdad(key, func=lambda x: x - 1),
                Invertd(key, transform=preprocessing, orig_keys=key),
            ]
        )

        item = {key: img}
        pre = preprocessing(item)
        post = postprocessing(pre)

        self.assertIsNotNone(post)
        self.assertIn(key, post)

    def test_invertd_group_isolation(self):
        """Test that groups correctly isolate transforms from different Compose instances."""
        img, _ = create_test_image_2d(60, 60, 2, 10, num_seg_classes=2)
        img = MetaTensor(img, meta={"original_channel_dim": float("nan"), "pixdim": [1.0, 1.0, 1.0]})
        key = "image"

        # First preprocessing
        preprocessing1 = Compose([EnsureChannelFirstd(key), Spacingd(key, pixdim=[2.0, 2.0])])

        # Second preprocessing (different pipeline)
        preprocessing2 = Compose([Spacingd(key, pixdim=[1.5, 1.5])])

        item = {key: img}
        pre1 = preprocessing1(item)

        # Verify group IDs are in applied_operations
        self.assertTrue(len(pre1[key].applied_operations) > 0)
        group1 = pre1[key].applied_operations[0].get("group")
        self.assertIsNotNone(group1)
        self.assertEqual(group1, str(id(preprocessing1)))

        # Apply second preprocessing
        pre2 = preprocessing2(pre1)
        self.assertTupleEqual(pre2[key].shape, (1, 40, 40))

        # Should have operations from both pipelines with different groups
        groups = [op.get("group") for op in pre2[key].applied_operations]
        preprocessing1_group = str(id(preprocessing1))
        preprocessing2_group = str(id(preprocessing2))
        self.assertIn(preprocessing1_group, groups)
        self.assertIn(preprocessing2_group, groups)
        self.assertEqual(groups.count(preprocessing1_group), 1)
        self.assertEqual(groups.count(preprocessing2_group), 1)

        # Inverting preprocessing1 should only invert its transforms
        inverter = Invertd(key, transform=preprocessing1, orig_keys=key)
        inverted = inverter(pre2)
        self.assertIsNotNone(inverted)
        self.assertTupleEqual(inverted[key].shape, (1, 60, 60))

    def test_compose_inverse_with_groups(self):
        """Test that Compose.inverse() works correctly with automatic group tracking."""
        img, _ = create_test_image_2d(60, 60, 2, 10, num_seg_classes=2)
        img = MetaTensor(img, meta={"original_channel_dim": float("nan"), "pixdim": [1.0, 1.0, 1.0]})
        key = "image"

        # Create a preprocessing pipeline
        preprocessing = Compose([EnsureChannelFirstd(key), Spacingd(key, pixdim=[2.0, 2.0])])

        # Apply preprocessing
        item = {key: img}
        pre = preprocessing(item)

        # Call inverse() directly on the Compose object
        inverted = preprocessing.inverse(pre)

        # Should successfully invert
        self.assertIsNotNone(inverted)
        self.assertIn(key, inverted)
        # Shape should be restored after inversion
        self.assertEqual(inverted[key].shape[1:], img.shape)

    def test_compose_inverse_with_postprocessing_groups(self):
        """Test Compose.inverse() when data has been through multiple pipelines with different groups."""
        img, _ = create_test_image_2d(60, 60, 2, 10, num_seg_classes=2)
        img = MetaTensor(img, meta={"original_channel_dim": float("nan"), "pixdim": [1.0, 1.0, 1.0]})
        key = "image"

        # Preprocessing pipeline
        preprocessing = Compose([EnsureChannelFirstd(key), Spacingd(key, pixdim=[2.0, 2.0])])

        # Postprocessing pipeline (different group)
        postprocessing = Compose([Lambdad(key, func=lambda x: x * 2)])

        # Apply both pipelines
        item = {key: img}
        pre = preprocessing(item)
        post = postprocessing(pre)

        # Now call inverse() directly on preprocessing
        # This tests that inverse() can handle data that has transforms from multiple groups
        # This WILL fail because applied_operations contains postprocessing transforms
        # and inverse() doesn't do group filtering (only Invertd does)
        with self.assertRaises(RuntimeError):
            preprocessing.inverse(post)

    def test_mixed_invertd_and_compose_inverse(self):
        """Test mixing Invertd (with group filtering) and Compose.inverse() (without filtering)."""
        img, _ = create_test_image_2d(60, 60, 2, 10, num_seg_classes=2)
        img = MetaTensor(img, meta={"original_channel_dim": float("nan"), "pixdim": [1.0, 1.0, 1.0]})
        key = "image"

        # First pipeline
        pipeline1 = Compose([EnsureChannelFirstd(key), Spacingd(key, pixdim=[2.0, 2.0])])

        # Apply first pipeline
        item = {key: img}
        result1 = pipeline1(item)

        # Use Compose.inverse() directly - should work fine
        inverted1 = pipeline1.inverse(result1)
        self.assertIsNotNone(inverted1)
        self.assertEqual(inverted1[key].shape[1:], img.shape)

        # Now apply pipeline again and use Invertd
        result2 = pipeline1(item)
        inverter = Invertd(key, transform=pipeline1, orig_keys=key)
        inverted2 = inverter(result2)
        self.assertIsNotNone(inverted2)


if __name__ == "__main__":
    unittest.main()

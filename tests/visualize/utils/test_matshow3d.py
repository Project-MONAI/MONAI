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

import tempfile
import unittest

import numpy as np

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandSpatialCropSamplesd,
    RepeatChanneld,
    ScaleIntensityd,
)
from monai.utils import optional_import
from monai.visualize.utils import matshow3d
from tests.test_utils import SkipIfNoModule, TESTING_DATA_PATH

compare_images, _ = optional_import("matplotlib.testing.compare", name="compare_images")
plt, has_pyplot = optional_import("matplotlib", name="pyplot")

TOL = 5.0  # FIXME: Matplotlib changed around 3.11 to render differently, high tolerance needed for compatibility

KEYS = "image"

BASE_XFORMS = [
    LoadImaged(keys=KEYS),
    EnsureChannelFirstd(keys=KEYS, channel_dim="no_channel"),
    ScaleIntensityd(keys=KEYS),
]

INPUT_IMAGE = TESTING_DATA_PATH / "anatomical.nii"


@SkipIfNoModule("matplotlib")
class TestMatshow3d(unittest.TestCase):
    def test_3d(self):
        xforms = Compose(BASE_XFORMS)
        ims = xforms({KEYS: INPUT_IMAGE})

        fig = plt.figure()  # external figure
        fig, _ = matshow3d(ims[KEYS], fig=fig, figsize=(2, 2), frames_per_row=5, every_n=2, frame_dim=-1, show=False)

        with tempfile.TemporaryDirectory() as tempdir:
            tempimg = f"{tempdir}/matshow3d_test.png"
            fig.savefig(tempimg)
            comp = compare_images(TESTING_DATA_PATH / "matshow3d_test.png", tempimg, TOL)
            self.assertIsNone(comp, f"compare_images result: {comp}")  # None indicates test passed

        _, axes = plt.subplots()
        matshow3d(ims[KEYS], fig=axes, figsize=(2, 2), frames_per_row=5, every_n=2, frame_dim=-1, show=False)

    def test_samples(self):
        xforms = Compose(
            [*BASE_XFORMS, RandSpatialCropSamplesd(keys=KEYS, roi_size=(8, 8, 5), random_size=True, num_samples=10)]
        )
        xforms.set_random_state(0)
        ims = [im[KEYS] for im in xforms({KEYS: INPUT_IMAGE})]
        fig, mat = matshow3d(ims, title=f"testing {KEYS}", figsize=(2, 2), frames_per_row=5, every_n=2, show=False)
        self.assertEqual(mat.dtype, np.float32)

        with tempfile.TemporaryDirectory() as tempdir:
            tempimg = f"{tempdir}/matshow3d_patch_test.png"
            fig.savefig(tempimg)
            comp = compare_images(TESTING_DATA_PATH / "matshow3d_patch_test.png", tempimg, TOL, in_decorator=True)
            if comp:
                # FIXME: tolerances are too high after changes to Matplotlib around version 3.11
                np.testing.assert_allclose(
                    comp["rms"], 30.786983, atol=3, rtol=0.1, err_msg=f"compare_images result: {comp}"
                )

    def test_3d_rgb(self):
        xforms = Compose([*BASE_XFORMS, RepeatChanneld(keys=KEYS, repeats=3)])  # RepeatChanneld makes a RGB color image
        ims = xforms({KEYS: INPUT_IMAGE})

        fig = plt.figure()  # external figure
        fig, _ = matshow3d(
            volume=ims[KEYS],
            fig=fig,
            figsize=(2, 2),
            frames_per_row=5,
            every_n=2,
            frame_dim=-1,
            channel_dim=0,
            fill_value=0,
            show=False,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            tempimg = f"{tempdir}/matshow3d_rgb_test.png"
            fig.savefig(tempimg)
            comp = compare_images(TESTING_DATA_PATH / "matshow3d_rgb_test.png", tempimg, TOL)
            self.assertIsNone(comp, f"compare_images result: {comp}")  # None indicates test passed


if __name__ == "__main__":
    unittest.main()

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

import os
import tempfile
import unittest

import numpy as np

from monai.transforms import AddChanneld, Compose, LoadImaged, RandSpatialCropSamplesd, RepeatChanneld, ScaleIntensityd
from monai.utils import optional_import
from monai.visualize.utils import matshow3d
from tests.utils import SkipIfNoModule

compare_images, _ = optional_import("matplotlib.testing.compare", name="compare_images")
pyplot, has_pyplot = optional_import("matplotlib", name="pyplot")


@SkipIfNoModule("matplotlib")
class TestMatshow3d(unittest.TestCase):
    def test_3d(self):
        testing_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")
        keys = "image"
        xforms = Compose([LoadImaged(keys=keys), AddChanneld(keys=keys), ScaleIntensityd(keys=keys)])
        image_path = os.path.join(testing_dir, "anatomical.nii")
        ims = xforms({keys: image_path})

        fig = pyplot.figure()  # external figure
        fig, _ = matshow3d(ims[keys], fig=fig, figsize=(2, 2), frames_per_row=5, every_n=2, frame_dim=-1, show=False)

        with tempfile.TemporaryDirectory() as tempdir:
            tempimg = f"{tempdir}/matshow3d_test.png"
            fig.savefig(tempimg)
            comp = compare_images(f"{testing_dir}/matshow3d_test.png", tempimg, 5e-2)
            self.assertIsNone(comp, f"value of comp={comp}")  # None indicates test passed

        _, axes = pyplot.subplots()
        matshow3d(ims[keys], fig=axes, figsize=(2, 2), frames_per_row=5, every_n=2, frame_dim=-1, show=False)

    def test_samples(self):
        testing_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")
        keys = "image"
        xforms = Compose(
            [
                LoadImaged(keys=keys),
                AddChanneld(keys=keys),
                ScaleIntensityd(keys=keys),
                RandSpatialCropSamplesd(keys=keys, roi_size=(8, 8, 5), random_size=True, num_samples=10),
            ]
        )
        image_path = os.path.join(testing_dir, "anatomical.nii")
        xforms.set_random_state(0)
        ims = xforms({keys: image_path})
        fig, mat = matshow3d(
            [im[keys] for im in ims], title=f"testing {keys}", figsize=(2, 2), frames_per_row=5, every_n=2, show=False
        )
        self.assertTrue(mat.dtype == np.float32)

        with tempfile.TemporaryDirectory() as tempdir:
            tempimg = f"{tempdir}/matshow3d_patch_test.png"
            fig.savefig(tempimg)
            comp = compare_images(f"{testing_dir}/matshow3d_patch_test.png", tempimg, 5e-2, in_decorator=True)
            if comp:
                print("not none comp: ", comp)  # matplotlib 3.2.2
                np.testing.assert_allclose(comp["rms"], 30.786983, atol=1e-3, rtol=1e-3)
            else:
                self.assertIsNone(comp, f"value of comp={comp}")  # None indicates test passed

    def test_3d_rgb(self):
        testing_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")
        keys = "image"
        xforms = Compose(
            [
                LoadImaged(keys=keys),
                AddChanneld(keys=keys),
                ScaleIntensityd(keys=keys),
                # change to RGB color image
                RepeatChanneld(keys=keys, repeats=3),
            ]
        )
        image_path = os.path.join(testing_dir, "anatomical.nii")
        ims = xforms({keys: image_path})

        fig = pyplot.figure()  # external figure
        fig, _ = matshow3d(
            volume=ims[keys],
            fig=fig,
            figsize=(2, 2),
            frames_per_row=5,
            every_n=2,
            frame_dim=-1,
            channel_dim=0,
            show=False,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            tempimg = f"{tempdir}/matshow3d_rgb_test.png"
            fig.savefig(tempimg)
            comp = compare_images(f"{testing_dir}/matshow3d_rgb_test.png", tempimg, 5e-2)
            self.assertIsNone(comp, f"value of comp={comp}")  # None indicates test passed


if __name__ == "__main__":
    unittest.main()

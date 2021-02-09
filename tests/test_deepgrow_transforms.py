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

from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    AddGuidanceSignald,
    AddInitialSeedPointd,
    Fetch2DSliced,
    FindDiscrepancyRegionsd,
    ResizeGuidanced,
    RestoreCroppedLabeld,
    SpatialCropGuidanced,
)
from monai.transforms import AddChanneld

DATA = {
    "image": np.array([[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]]),
    "label": np.array([[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]),
    "image_meta_dict": {},
    "label_meta_dict": {},
    "foreground": [[2, 2, 0]],
    "background": [],
}


class TestTransforms(unittest.TestCase):
    def test_addinitialseedpointd_addguidancesignald(self):
        result = AddChanneld(keys=("image", "label"))(DATA.copy())
        result = AddInitialSeedPointd(label="label", guidance="guidance", sids="sids")(result)
        assert len(result["guidance"])

        result = AddGuidanceSignald(image="image", guidance="guidance")(result)
        assert result["image"].shape == (3, 1, 5, 5)

    def test_finddiscrepancyregionsd(self):
        result = DATA.copy()
        result["pred"] = np.array(
            [[[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]]
        )
        result = AddChanneld(keys=("image", "label", "pred"))(result)
        result = FindDiscrepancyRegionsd(label="label", pred="pred", discrepancy="discrepancy")(result)
        assert np.sum(result["discrepancy"]) > 0

    def test_inference(self):
        result = DATA.copy()
        result["image_meta_dict"]["spatial_shape"] = (5, 5, 1)
        result["image_meta_dict"]["original_affine"] = (0, 0)

        result = AddGuidanceFromPointsd(
            ref_image="image", guidance="guidance", foreground="foreground", background="background", dimensions=2
        )(result)
        assert len(result["guidance"][0][0]) == 2

        result = Fetch2DSliced(keys="image", guidance="guidance")(result)
        assert result["image"].shape == (5, 5)

        result = AddChanneld(keys="image")(result)

        result = SpatialCropGuidanced(keys="image", guidance="guidance", spatial_size=(4, 4))(result)
        assert result["image"].shape == (1, 4, 4)

        result = ResizeGuidanced(guidance="guidance", ref_image="image")(result)

        result["pred"] = np.random.randint(0, 2, size=(1, 4, 4))
        result = RestoreCroppedLabeld(keys="pred", ref_image="image", mode="nearest")(result)
        assert result["pred"].shape == (1, 5, 5)


if __name__ == "__main__":
    unittest.main()

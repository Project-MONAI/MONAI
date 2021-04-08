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
from ignite.engine import Engine

from monai.data import CacheDataset, DataLoader, create_test_image_3d
from monai.handlers import TransformInverter
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandAffined,
    RandAxisFlipd,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    ResizeWithPadOrCropd,
    ToTensord,
)
from tests.utils import make_nifti_image

KEYS = ["image", "label"]


class TestTransformInverter(unittest.TestCase):
    def test_invert(self):
        im_fname, seg_fname = [make_nifti_image(i) for i in create_test_image_3d(101, 100, 107)]
        transform = Compose(
            [
                LoadImaged(KEYS),
                AddChanneld(KEYS),
                RandFlipd(KEYS, prob=0.5, spatial_axis=[1, 2]),
                RandAxisFlipd(KEYS, prob=0.5),
                RandRotate90d(KEYS, spatial_axes=(1, 2)),
                RandZoomd(KEYS, prob=0.5, min_zoom=0.5, max_zoom=1.1, keep_size=True),
                RandRotated(KEYS, prob=0.5, range_x=np.pi),
                RandAffined(KEYS, prob=0.5, rotate_range=np.pi),
                ResizeWithPadOrCropd(KEYS, 100),
                ToTensord(KEYS),
            ]
        )
        data = [{"image": im_fname, "label": seg_fname} for _ in range(12)]

        # num workers = 0 for mac or gpu transforms
        num_workers = 0 if sys.platform == "darwin" or torch.cuda.is_available() else 2

        dataset = CacheDataset(data, transform=transform, progress=False)
        loader = DataLoader(dataset, num_workers=num_workers, batch_size=5)

        # set up engine
        def _train_func(engine, batch):
            self.assertTupleEqual(batch["image"].shape[1:], (1, 100, 100, 100))
            return batch

        engine = Engine(_train_func)

        # set up testing handler
        TransformInverter(transform=transform, loader=loader, output_key="image").attach(engine)

        engine.run(loader, max_epochs=1)
        self.assertTupleEqual(engine.state.output["image"].shape, (2, 1, 100, 100, 100))
        for i in engine.state.output["image_inverted"]:
            self.assertTupleEqual(i.shape, (1, 100, 101, 107))


if __name__ == "__main__":
    unittest.main()

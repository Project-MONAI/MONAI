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

from monai.data import (
    create_test_image_2d,
    CacheDataset,
    DataLoader,
)
from monai.transforms import (
    Compose,
    RandAffined,
    Spacingd
)


class TestTransformsWCacheDatasetAndPersistentWorkers(unittest.TestCase):
    def test_duplicate_transforms(self):
        im, _ = create_test_image_2d(128, 128, num_seg_classes=1, channel_dim=0)
        data = [{"img": im} for _ in range(2)]

        # at least 1 deterministic followed by at least 1 random
        transform = Compose([
            Spacingd("img", pixdim=(1, 1)),
            RandAffined("img", prob=1.0),
        ])

        # cachedataset and data loader w persistent_workers
        train_ds = CacheDataset(data, transform, cache_num=1)
        train_loader = DataLoader(train_ds, num_workers=2, persistent_workers=True)

        b1 = next(iter(train_loader))
        b2 = next(iter(train_loader))

        self.assertEqual(len(b1["img_transforms"]), len(b2["img_transforms"]))


if __name__ == "__main__":
    # unittest.main()
    a = TestTransformsWCacheDatasetAndPersistentWorkers()
    a.test_duplicate_transforms()

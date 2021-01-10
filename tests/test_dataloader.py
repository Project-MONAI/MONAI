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

from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, DataStatsd, SimulateDelayd


class TestDataLoader(unittest.TestCase):
    def test_values(self):
        datalist = [
            {"image": "spleen_19.nii.gz", "label": "spleen_label_19.nii.gz"},
            {"image": "spleen_31.nii.gz", "label": "spleen_label_31.nii.gz"},
        ]
        transform = Compose(
            [
                DataStatsd(keys=["image", "label"], data_shape=False, value_range=False, data_value=True),
                SimulateDelayd(keys=["image", "label"], delay_time=0.1),
            ]
        )
        dataset = CacheDataset(data=datalist, transform=transform, cache_rate=0.5, cache_num=1)
        n_workers = 0 if sys.platform == "win32" else 2
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=n_workers)
        for d in dataloader:
            self.assertEqual(d["image"][0], "spleen_19.nii.gz")
            self.assertEqual(d["image"][1], "spleen_31.nii.gz")
            self.assertEqual(d["label"][0], "spleen_label_19.nii.gz")
            self.assertEqual(d["label"][1], "spleen_label_31.nii.gz")


if __name__ == "__main__":
    unittest.main()

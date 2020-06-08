# Copyright 2020 MONAI Consortium
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
from monai.data import CacheDataset, DataLoader
from monai.transforms import DataStatsd, SimulateDelayd, Compose


class TestDataLoader(unittest.TestCase):
    def test_values(self):
        datalist = [
            {"data": "spleen_19.nii.gz", "label": "spleen_label_19.nii.gz"},
            {"data": "spleen_31.nii.gz", "label": "spleen_label_31.nii.gz"},
        ]
        transform = Compose(
            [
                DataStatsd(keys=["data", "label"], data_shape=False, intensity_range=False, data_value=True),
                SimulateDelayd(keys=["data", "label"], delay_time=0.1),
            ]
        )
        dataset = CacheDataset(data=datalist, transform=transform, cache_rate=0.5, cache_num=1)
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=2)
        for d in dataloader:
            self.assertEqual(d["data"][0], "spleen_19.nii.gz")
            self.assertEqual(d["data"][1], "spleen_31.nii.gz")
            self.assertEqual(d["label"][0], "spleen_label_19.nii.gz")
            self.assertEqual(d["label"][1], "spleen_label_31.nii.gz")


if __name__ == "__main__":
    unittest.main()

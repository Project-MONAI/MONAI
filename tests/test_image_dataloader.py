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
import os
import shutil
import tempfile
from parameterized import parameterized
from monai.data import ImageDataLoader
from monai.transforms import DataStatsd, SimulateDelayd, Compose

TEST_CASE_1 = [0, 0, None]

TEST_CASE_2 = [0.5, 1, None]

TEST_CASE_3 = [0, 0, "cache_data"]


class TestImageDataLoader(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_values(self, cache_rate, cache_num, cache_dir):
        tempdir = tempfile.mkdtemp()
        if cache_dir is not None:
            cache_dir = os.path.join(tempdir, cache_dir)
        datalist = [
            {"image": "spleen_19.nii.gz", "label": "spleen_label_19.nii.gz"},
            {"image": "spleen_31.nii.gz", "label": "spleen_label_31.nii.gz"},
        ]
        transform = Compose(
            [
                DataStatsd(keys=["image", "label"], data_shape=False, intensity_range=False, data_value=True),
                SimulateDelayd(keys=["image", "label"], delay_time=0.1),
            ]
        )
        dataloader = ImageDataLoader(
            datalist=datalist,
            transform=transform,
            batch_size=2,
            cache_rate=cache_rate,
            cache_num=cache_num,
            cache_dir=cache_dir,
        )
        for d in dataloader:
            self.assertEqual(d["image"][0], "spleen_19.nii.gz")
            self.assertEqual(d["image"][1], "spleen_31.nii.gz")
            self.assertEqual(d["label"][0], "spleen_label_19.nii.gz")
            self.assertEqual(d["label"][1], "spleen_label_31.nii.gz")
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()

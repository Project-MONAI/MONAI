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

import csv
import os
import tempfile
import unittest

import numpy as np
import torch

from monai.data import CSVSaver
from monai.transforms import Compose, CopyItemsd, SaveClassificationd


class TestSaveClassificationd(unittest.TestCase):
    def test_saved_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            data = [
                {
                    "pred": torch.zeros(8),
                    "image_meta_dict": {"filename_or_obj": ["testfile" + str(i) for i in range(8)]},
                },
                {
                    "pred": torch.zeros(8),
                    "image_meta_dict": {"filename_or_obj": ["testfile" + str(i) for i in range(8, 16)]},
                },
            ]

            saver = CSVSaver(output_dir=tempdir, filename="predictions.csv", overwrite=False)
            # set up test transforms
            saver_classification = SaveClassificationd(
                keys="pred",
                saver=None,
                output_dir=tempdir,
                filename="predictions.csv",
                overwrite=False,
            )
            post_trans = Compose(
                [
                    CopyItemsd(keys="image_meta_dict", times=1, names="pred_meta_dict"),
                    # 1st saver saves data into CSV file
                    saver_classification,
                    # 2nd saver will not save new data due to `overwrite=False`
                    SaveClassificationd(keys="pred", saver=saver),
                ]
            )
            # simulate inference 2 iterations
            for d in data:
                post_trans(d)
            # write into CSV file
            saver_classification.get_saver().finalize()
            saver.finalize()

            filepath = os.path.join(tempdir, "predictions.csv")
            self.assertTrue(os.path.exists(filepath))
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                i = 0
                for row in reader:
                    self.assertEqual(row[0], "testfile" + str(i))
                    self.assertEqual(np.array(row[1:]).astype(np.float32), 0.0)
                    i += 1
                self.assertEqual(i, 16)


if __name__ == "__main__":
    unittest.main()

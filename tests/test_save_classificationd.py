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
                {
                    "pred": torch.zeros(8),
                    "image_meta_dict": {"filename_or_obj": ["testfile" + str(i) for i in range(16, 24)]},
                },
            ]

            saver = CSVSaver(output_dir=tempdir, filename="predictions2.csv", overwrite=False, flush=False)
            # set up test transforms
            post_trans = Compose(
                [
                    CopyItemsd(keys="image_meta_dict", times=1, names="pred_meta_dict"),
                    # 1st saver saves data into CSV file
                    SaveClassificationd(
                        keys="pred",
                        saver=None,
                        meta_keys=None,
                        output_dir=tempdir,
                        filename="predictions1.csv",
                        overwrite=True,
                    ),
                    # 2rd saver only saves data into the cache, manually finalize later
                    SaveClassificationd(keys="pred", saver=saver, meta_key_postfix="meta_dict"),
                ]
            )
            # simulate inference 2 iterations
            post_trans(data[0])
            post_trans(data[1])
            # write into CSV file
            saver.finalize()

            # 3rd saver will not delete previous data due to `overwrite=False`
            SaveClassificationd(
                keys="pred",
                saver=None,
                meta_keys="image_meta_dict",  # specify meta key, so no need to copy anymore
                output_dir=tempdir,
                filename="predictions1.csv",
                overwrite=False,
            )(data[2])

            def _test_file(filename, count):
                filepath = os.path.join(tempdir, filename)
                self.assertTrue(os.path.exists(filepath))
                with open(filepath, "r") as f:
                    reader = csv.reader(f)
                    i = 0
                    for row in reader:
                        self.assertEqual(row[0], "testfile" + str(i))
                        self.assertEqual(np.array(row[1:]).astype(np.float32), 0.0)
                        i += 1
                    self.assertEqual(i, count)

            _test_file("predictions1.csv", 24)
            _test_file("predictions2.csv", 16)


if __name__ == "__main__":
    unittest.main()

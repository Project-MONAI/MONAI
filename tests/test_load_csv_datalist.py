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

from collections import OrderedDict
import os
import tempfile
import unittest

from monai.data import load_csv_datalist


class TestLoadCSVDatalist(unittest.TestCase):
    def test_values(self):
        with tempfile.TemporaryDirectory() as tempdir:
            test_data1 = [
                ["subject_id", "label", "image", "ehr_0", "ehr_1", "ehr_2"],
                ["s000000", 5, "./imgs/s000000.png", 2.007843256, 2.29019618, 2.054902077],
                ["s000001", 0, "./imgs/s000001.png", 6.839215755, 6.474509716, 5.862744808],
                ["s000002", 4, "./imgs/s000002.png", 3.772548914, 4.211764812, 4.635294437],
                ["s000003", 1, "./imgs/s000003.png", 3.333333254, 3.235294342, 3.400000095],
                ["s000004", 9, "./imgs/s000004.png", 6.427451134, 6.254901886, 5.976470947],
            ]
            test_data2 = [

            ]

            def prepare_csv_file(data, filepath):
                with open(filepath, "a") as f:
                    for d in data:
                        f.write((",".join([str(i) for i in d])) + "\n")

            filepath1 = os.path.join(tempdir, "test_data1.csv")
            prepare_csv_file(test_data1, filepath1)

            # load single CSV file
            result = load_csv_datalist(filepath1)
            self.assertDictEqual(
                {k: round(v, 4) if not isinstance(v, str) else v for k, v in result[2].items()},
                {
                    "subject_id": "s000002",
                    "label": 4,
                    "image": "./imgs/s000002.png",
                    "ehr_0": 3.7725,
                    "ehr_1": 4.2118,
                    "ehr_2": 4.6353,
                },
            )

if __name__ == "__main__":
    unittest.main()

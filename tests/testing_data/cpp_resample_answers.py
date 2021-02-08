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

Expected_1D_BP_fwd = []
pwd = os.path.dirname(os.path.abspath(__file__))  # current file's location
with open(os.path.join(pwd, "cpp_resample_answers.txt")) as f:
    res_reader = csv.reader(f, delimiter=",")
    for r in res_reader:
        res_row = []
        for item in r:
            if item.strip().startswith("#"):
                continue
            res_row.append(float(item))
        Expected_1D_BP_fwd.append(res_row)

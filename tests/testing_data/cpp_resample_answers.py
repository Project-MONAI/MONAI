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
import warnings
from typing import List, Optional


def _read_testing_data_answers(fname: Optional[str] = None, delimiter=",") -> List:
    answers: List = []
    if not fname:
        return answers
    # read answers from directory of the current file
    pwd = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(pwd, fname)
    if not os.path.isfile(filename):
        warnings.warn("test data {} not found.".format(filename))
        return answers
    with open(filename) as f:
        res_reader = csv.reader(f, delimiter=delimiter)
        for r in res_reader:
            res_row = []
            for item in r:
                if item.strip().startswith("#"):
                    continue  # allow for some simple comments in the file
                res_row.append(float(item))
            answers.append(res_row)
    return answers


Expected_1D_GP_fwd: List = _read_testing_data_answers(fname="1D_BP_fwd.txt")
Expected_1D_GP_bwd: List = _read_testing_data_answers(fname="1D_BP_bwd.txt")

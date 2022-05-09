# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import DataStats

TEST_CASE_1 = [
    {
        "prefix": "test data",
        "data_type": False,
        "data_shape": False,
        "value_range": False,
        "data_value": False,
        "additional_info": None,
        "name": "DataStats",
    },
    np.array([[0, 1], [1, 2]]),
    "test data statistics:",
]

TEST_CASE_2 = [
    {
        "prefix": "test data",
        "data_type": True,
        "data_shape": False,
        "value_range": False,
        "data_value": False,
        "additional_info": None,
        "name": "DataStats",
    },
    np.array([[0, 1], [1, 2]]),
    "test data statistics:\nType: <class 'numpy.ndarray'>",
]

TEST_CASE_3 = [
    {
        "prefix": "test data",
        "data_type": True,
        "data_shape": True,
        "value_range": False,
        "data_value": False,
        "additional_info": None,
        "name": "DataStats",
    },
    np.array([[0, 1], [1, 2]]),
    "test data statistics:\nType: <class 'numpy.ndarray'>\nShape: (2, 2)",
]

TEST_CASE_4 = [
    {
        "prefix": "test data",
        "data_type": True,
        "data_shape": True,
        "value_range": True,
        "data_value": False,
        "additional_info": None,
        "name": "DataStats",
    },
    np.array([[0, 1], [1, 2]]),
    "test data statistics:\nType: <class 'numpy.ndarray'>\nShape: (2, 2)\nValue range: (0, 2)",
]

TEST_CASE_5 = [
    {
        "prefix": "test data",
        "data_type": True,
        "data_shape": True,
        "value_range": True,
        "data_value": True,
        "additional_info": None,
        "name": "DataStats",
    },
    np.array([[0, 1], [1, 2]]),
    "test data statistics:\nType: <class 'numpy.ndarray'>\nShape: (2, 2)\nValue range: (0, 2)\nValue: [[0 1]\n [1 2]]",
]

TEST_CASE_6 = [
    {
        "prefix": "test data",
        "data_type": True,
        "data_shape": True,
        "value_range": True,
        "data_value": True,
        "additional_info": np.mean,
        "name": "DataStats",
    },
    np.array([[0, 1], [1, 2]]),
    (
        "test data statistics:\nType: <class 'numpy.ndarray'>\nShape: (2, 2)\n"
        "Value range: (0, 2)\nValue: [[0 1]\n [1 2]]\nAdditional info: 1.0"
    ),
]

TEST_CASE_7 = [
    {
        "prefix": "test data",
        "data_type": True,
        "data_shape": True,
        "value_range": True,
        "data_value": True,
        "additional_info": lambda x: torch.mean(x.float()),
        "name": "DataStats",
    },
    torch.tensor([[0, 1], [1, 2]]).to("cuda" if torch.cuda.is_available() else "cpu"),
    (
        "test data statistics:\nType: <class 'torch.Tensor'>\nShape: torch.Size([2, 2])\nValue range: (0, 2)\n"
        "Value: tensor([[0, 1],\n        [1, 2]])\nAdditional info: 1.0"
    ),
]

TEST_CASE_8 = [
    np.array([[0, 1], [1, 2]]),
    "test data statistics:\nType: <class 'numpy.ndarray'> int64\nShape: (2, 2)\nValue range: (0, 2)\n"
    "Value: [[0 1]\n [1 2]]\nAdditional info: 1.0\n",
]


class TestDataStats(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7])
    def test_value(self, input_param, input_data, expected_print):
        transform = DataStats(**input_param)
        _ = transform(input_data)

    @parameterized.expand([TEST_CASE_8])
    def test_file(self, input_data, expected_print):
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_data_stats.log")
            handler = logging.FileHandler(filename, mode="w")
            handler.setLevel(logging.INFO)
            name = "DataStats"
            logger = logging.getLogger(name)
            logger.addHandler(handler)
            input_param = {
                "prefix": "test data",
                "data_type": True,
                "data_shape": True,
                "value_range": True,
                "data_value": True,
                "additional_info": np.mean,
                "name": name,
            }
            transform = DataStats(**input_param)
            _ = transform(input_data)
            for h in logger.handlers[:]:
                h.close()
                logger.removeHandler(h)
            with open(filename) as f:
                content = f.read()
            if sys.platform != "win32":
                self.assertEqual(content, expected_print)


if __name__ == "__main__":
    unittest.main()

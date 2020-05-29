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
import logging
import tempfile
import numpy as np
from parameterized import parameterized
from monai.transforms import DataStatsd

TEST_CASE_1 = [
    {
        "keys": "img",
        "prefix": "test data",
        "data_shape": False,
        "intensity_range": False,
        "data_value": False,
        "additional_info": None,
    },
    {"img": np.array([[0, 1], [1, 2]])},
    "test data statistics:",
]

TEST_CASE_2 = [
    {
        "keys": "img",
        "prefix": "test data",
        "data_shape": True,
        "intensity_range": False,
        "data_value": False,
        "additional_info": None,
    },
    {"img": np.array([[0, 1], [1, 2]])},
    "test data statistics:\nShape: (2, 2)",
]

TEST_CASE_3 = [
    {
        "keys": "img",
        "prefix": "test data",
        "data_shape": True,
        "intensity_range": True,
        "data_value": False,
        "additional_info": None,
    },
    {"img": np.array([[0, 1], [1, 2]])},
    "test data statistics:\nShape: (2, 2)\nIntensity range: (0, 2)",
]

TEST_CASE_4 = [
    {
        "keys": "img",
        "prefix": "test data",
        "data_shape": True,
        "intensity_range": True,
        "data_value": True,
        "additional_info": None,
    },
    {"img": np.array([[0, 1], [1, 2]])},
    "test data statistics:\nShape: (2, 2)\nIntensity range: (0, 2)\nValue: [[0 1]\n [1 2]]",
]

TEST_CASE_5 = [
    {
        "keys": "img",
        "prefix": "test data",
        "data_shape": True,
        "intensity_range": True,
        "data_value": True,
        "additional_info": lambda x: np.mean(x),
    },
    {"img": np.array([[0, 1], [1, 2]])},
    "test data statistics:\nShape: (2, 2)\nIntensity range: (0, 2)\nValue: [[0 1]\n [1 2]]\nAdditional info: 1.0",
]

TEST_CASE_6 = [
    {
        "keys": ("img", "affine"),
        "prefix": ("image", "affine"),
        "data_shape": True,
        "intensity_range": (True, False),
        "data_value": (False, True),
        "additional_info": (lambda x: np.mean(x), None),
    },
    {"img": np.array([[0, 1], [1, 2]]), "affine": np.eye(2, 2)},
    "affine statistics:\nShape: (2, 2)\nValue: [[1. 0.]\n [0. 1.]]",
]

TEST_CASE_7 = [
    {"img": np.array([[0, 1], [1, 2]])},
    "test data statistics:\nShape: (2, 2)\nIntensity range: (0, 2)\nValue: [[0 1]\n [1 2]]\nAdditional info: 1.0\n",
]


class TestDataStatsd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_value(self, input_param, input_data, expected_print):
        transform = DataStatsd(**input_param)
        _ = transform(input_data)
        self.assertEqual(transform.printer.output, expected_print)

    @parameterized.expand([TEST_CASE_7])
    def test_file(self, input_data, expected_print):
        tempdir = tempfile.mkdtemp()
        filename = os.path.join(tempdir, "test_stats.log")
        handler = logging.FileHandler(filename, mode="w")
        input_param = {
            "keys": "img",
            "prefix": "test data",
            "data_shape": True,
            "intensity_range": True,
            "data_value": True,
            "additional_info": lambda x: np.mean(x),
            "logger_handler": handler,
        }
        transform = DataStatsd(**input_param)
        _ = transform(input_data)
        handler.stream.close()
        transform.printer._logger.removeHandler(handler)
        with open(filename, "r") as f:
            content = f.read()
            self.assertEqual(content, expected_print)
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()

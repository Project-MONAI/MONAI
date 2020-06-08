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
from collections import OrderedDict

import numpy as np
from parameterized import parameterized

from typing import Dict, Any, List
from typing_extensions import TypedDict

from monai.utils import validate_kwargs


class TestkwargsValidation(unittest.TestCase):
    def test_correct_results(self):
        mykwargs1: Dict[str, Any] = {"arg1": 1, "arg2": "two", "arg3": False}
        myargs1: List[Any] = []
        reference_args: OrderedDict = OrderedDict(
            {"arg1": "__from_input__", "arg2": "__from_input__", "arg3": "__from_input__", "defaultarg4": "good"}
        )

        # Valid input argument test 1
        required_output1: Dict[str, Any] = {"arg1": 1, "arg2": "two", "arg3": False, "defaultarg4": "good"}
        produced_args1: Dict[str, Any] = validate_kwargs(myargs1, mykwargs1, reference_args)
        self.assertDictEqual(required_output1, produced_args1)

        # Valid input argument test 2
        mykwargs2: Dict[str, Any] = {"arg1": 1, "arg2": "two", "arg3": False, "defaultarg4": "reallygood"}
        myargs2: List[Any] = []
        required_output2: Dict[str, Any] = {"arg1": 1, "arg2": "two", "arg3": False, "defaultarg4": "reallygood"}
        produced_args2: Dict[str, Any] = validate_kwargs(myargs2, mykwargs2, reference_args)
        self.assertDictEqual(required_output2, produced_args2)

        # Missing mandatory input argument
        mykwargs3: Dict[str, Any] = {"arg2": "two", "arg3": False}
        myargs3: List[Any] = []
        with self.assertRaises(KeyError):
            _: Dict[str, Any] = validate_kwargs(myargs3, mykwargs3, reference_args)

        # Unknown input argument
        mykwargs4: Dict[str, Any] = {"arg1": 1, "arg2": "two", "arg3": False, "arg100": None}
        myargs4: List[Any] = []
        with self.assertRaises(KeyError):
            _: Dict[str, Any] = validate_kwargs(myargs4, mykwargs4, reference_args)

        # Valid input argument test 2
        mykwargs5: Dict[str, Any] = {}
        myargs5: List[Any] = [1, "two", False, "extragood"]
        required_output2: Dict[str, Any] = {"arg1": 1, "arg2": "two", "arg3": False, "defaultarg4": "extragood"}
        produced_args2: Dict[str, Any] = validate_kwargs(myargs5, mykwargs5, reference_args)
        self.assertDictEqual(required_output2, produced_args2)


if __name__ == "__main__":
    unittest.main()

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

import unittest

import torch
from parameterized import parameterized
import numpy as np

from monai.metrics import MorphologicalHausdorffDistanceMetric

a = torch.zeros(20,20,20)
b = torch.zeros(20,20,20)
a[0,0,0]=1
b[0,0,10]=1
numbToLookFor= torch.ones(1)

TEST_CASES = [
    [[a,b,1.0,numbToLookFor], 10]

]
class TestHausdorffDistanceMorphological(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value(self, input_data, expected_value):
        [y_pred, y, percentt, numberToLookFor] = input_data
        hd_metric = MorphologicalHausdorffDistanceMetric(percent=percentt)
        result= hd_metric.compute_hausdorff_distance(y_pred,y,numberToLookFor )
        np.testing.assert_allclose(expected_value, result, rtol=1e-7)



if __name__ == "__main__":
    unittest.main()

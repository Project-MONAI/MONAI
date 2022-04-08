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

import numpy as np
import torch
from parameterized import parameterized

from monai.metrics import MorphologicalHausdorffDistanceMetric
from tests.utils import skip_if_no_cuda

device = torch.device("cuda")


### testing single points diffrent dims
#dim1
compare_values = torch.ones(1)
a = torch.zeros(100, 100, 100)
b = torch.zeros(100, 100, 100)
a[0, 0, 0] = 1
b[10, 0, 0] = 1

#dim2
a1 = torch.zeros(200, 200, 200)
b1 = torch.zeros(200, 200, 200)
a1[0, 0, 0] = 1
b1[0, 15, 0] = 1

#dim3
a2 = torch.zeros(400, 200, 300)
b2 = torch.zeros(400, 200, 300)
a2[0, 0, 10] = 1
b2[0, 0, 150] = 1

### testing whole llines and compare_values set to 2
compare_valuesB = torch.ones(1)
compare_valuesB[0]=2
a3 = torch.zeros(400, 200, 300)
b3 = torch.zeros(400, 200, 300)
a3[:, 0, 10] = 2
b3[:, 0, 150] = 2

a4 = torch.zeros(400, 200, 300)
b4 = torch.zeros(400, 200, 300)
a4[10, 0, :] = 2
b4[120, 0, :] = 2


a5 = torch.zeros(400, 200, 300)
b5 = torch.zeros(400, 200, 300)
a5[10, :, 0] = 2
b5[120, :, 0] = 2


## testing whole planes
a6 = torch.zeros(400, 200, 300)
b6 = torch.zeros(400, 200, 300)
a6[10, :, :] = 2
b6[120, :, :] = 2


a7 = torch.zeros(400, 200, 300)
b7 = torch.zeros(400, 200, 300)
a7[:, 0, :] = 2
b7[:,110, :] = 2

a8 = torch.zeros(400, 200, 300)
b8 = torch.zeros(400, 200, 300)
#a8[:, :, 20] = 2
#b8[:,:, 130] = 2


a8[1, 1, 20]= 2
b8[1,1, 130]= 2
a8[2, 2, 20]= 2
b8[2,2, 130]= 2

#multi points
a9 = torch.zeros(400, 200, 300)
b9 = torch.zeros(400, 200, 300)

a9[0, 20,0 ]= 2
a9[0, 0,30 ]= 2
a9[40, 0,0 ]= 2
b9[0,0,0 ]= 2

TEST_CASES = [
    [[a, b, 1.0, compare_values], 10]
    ,[[a1, b1, 1.0, compare_values], 15]
    ,[[a2, b2, 1.0, compare_values], 140]
    ,[[a3, b3, 1.0, compare_valuesB], 140]
    ,[[a4, b4, 1.0, compare_valuesB], 110]
    ,[[a5, b5, 1.0, compare_valuesB], 110]
    ,[[a6, b6, 1.0, compare_valuesB], 110]
    ,[[a7, b7, 1.0, compare_valuesB], 110]
    ,[[a8, b8, 1.0, compare_valuesB], 110]

    #testing robust
    ,[[a6, b6, 0.9, compare_valuesB], 110]
    ,[[a7, b7, 0.85, compare_valuesB], 110]
    ,[[a8, b8, 0.8, compare_valuesB], 110]
    #multi points
    ,[[a9, b9, 1.0, compare_valuesB], 40]
    ]


@skip_if_no_cuda
class TestHausdorffDistanceMorphological(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value(self, input_data, expected_value):
        [y_pred, y, percentt, compare_values] = input_data
        hd_metric = MorphologicalHausdorffDistanceMetric(percentt)
        result = hd_metric.compute_hausdorff_distance(y_pred.to(device), y.to(device), compare_values.to(device),True)
        np.testing.assert_allclose(expected_value, result, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()

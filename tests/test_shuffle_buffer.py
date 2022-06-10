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

import sys
import unittest

import numpy as np

from monai.data import DataLoader, ShuffleBuffer
from monai.utils import convert_data_type


class TestShuffleBuffer(unittest.TestCase):
    def test_shape(self):

        buffer = ShuffleBuffer([1, 2, 3, 4], seed=0)
        num_workers = 2 if sys.platform == "linux" else 0
        dataloader = DataLoader(dataset=buffer, batch_size=2, num_workers=num_workers)
        output = [convert_data_type(x, np.ndarray)[0] for x in dataloader]
        np.testing.assert_allclose(output, [[2, 1], [3, 4]])


if __name__ == "__main__":
    unittest.main()

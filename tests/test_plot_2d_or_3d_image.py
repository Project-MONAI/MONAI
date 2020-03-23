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

import glob
import os
import shutil
import unittest
from torch.utils.tensorboard import SummaryWriter
import torch
from monai.visualize.img2tensorboard import plot_2d_or_3d_image


class TestPlot2dOr3dImage(unittest.TestCase):

    def test_tb_image_shape(self):
        default_dir = os.path.join('.', 'runs')
        shutil.rmtree(default_dir, ignore_errors=True)

        plot_2d_or_3d_image(torch.zeros((1, 1, 10, 10)), 0, SummaryWriter())

        self.assertTrue(os.path.exists(default_dir))
        self.assertTrue(len(glob.glob(default_dir)) > 0)
        shutil.rmtree(default_dir)


if __name__ == '__main__':
    unittest.main()

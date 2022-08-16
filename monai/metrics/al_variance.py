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

import warnings
from typing import List, Union

import numpy as np
import torch

from monai.utils import MetricReduction, convert_data_type

from .metric import Metric, CumulativeIterationMetric


def variance_metric(self, input_image, threshold_value=0.0005):
    '''

    :param self:
    :param input_image: The Input image is Repeats, Channels, Volume, Height, Depth or Repeats, Channels, Height, Depth
    :param threshold_value:
    :return: A N-dimension spatial map or a single scalar value of sum/mean depending upon choice
    '''
    input_image = input_image.astype(dtype="float32")

    # Threshold values less than or equal to zero
    threshold = threshold_value
    input_image[input_image <= 0] = threshold

    vari = np.nanvar(input_image, axis=0)
    variance = np.sum(vari, axis=0)

    if self.dimension == 3:
        variance = np.expand_dims(variance, axis=0)
        variance = np.expand_dims(variance, axis=0)
    return variance
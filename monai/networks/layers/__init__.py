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

from .convutils import calculate_out_shape, gaussian_1d, polyval, same_padding, stride_minus_kernel_padding
from .drop_path import DropPath
from .factories import Act, Conv, Dropout, LayerFactory, Norm, Pad, Pool, split_args
from .filtering import BilateralFilter, PHLFilter
from .gmm import GaussianMixtureModel
from .simplelayers import (
    LLTM,
    ChannelPad,
    Flatten,
    GaussianFilter,
    HilbertTransform,
    MedianFilter,
    Reshape,
    SavitzkyGolayFilter,
    SkipConnection,
    apply_filter,
    median_filter,
    separable_filtering,
)
from .spatial_transforms import AffineTransform, grid_count, grid_grad, grid_pull, grid_push
from .utils import get_act_layer, get_dropout_layer, get_norm_layer, get_pool_layer
from .weight_init import _no_grad_trunc_normal_, trunc_normal_

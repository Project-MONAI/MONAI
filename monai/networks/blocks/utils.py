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

from typing import Dict, Optional, Tuple, Union

from monai.networks.layers.factories import Act, Norm, split_args
from monai.utils import has_option


def get_norm_layer(spatial_dims: int, norm: Union[Tuple, str], channels: Optional[int] = None):
    norm_name, norm_args = split_args(norm)
    norm_type = Norm[norm_name, spatial_dims]
    kw_args = dict(norm_args)
    if has_option(norm_type, "num_features") and "num_features" not in kw_args:
        kw_args["num_features"] = channels
    if has_option(norm_type, "num_channels") and "num_channels" not in kw_args:
        kw_args["num_channels"] = channels
    return norm_type(**kw_args)


def get_act_layer(act: Union[Tuple, str]):
    act_name, act_args = split_args(act)
    act_type = Act[act_name]
    return act_type(**act_args)

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

from typing import Any, Dict, Optional, Tuple, Union

import torch

from monai.networks.layers import Conv, get_pool_layer
from monai.networks.utils import look_up_named_module, set_named_module
from monai.utils import deprecated_arg


class NetAdapter(torch.nn.Module):
    """
    Wrapper to replace the last layer of model by convolutional layer or FC layer.
    This module expects the output of `model layers[0: -2]` is a feature map with shape [B, C, spatial dims],
    then replace the model's last two layers with an optional `pooling` and a `conv` or `linear` layer.

    Args:
        model: a PyTorch model, which can be both 2D and 3D models. typically, it can be a pretrained model
            in Torchvision, like: ``resnet18``, ``resnet34``, ``resnet50``, ``resnet101``, ``resnet152``, etc.
            more details: https://pytorch.org/vision/stable/models.html.
        num_classes: number of classes for the last classification layer. Default to 1.
        dim: number of supported spatial dimensions in the specified model, depends on the model implementation.
            default to 2 as most Torchvision models are for 2D image processing.
        in_channels: number of the input channels of last layer. if None, get it from `in_features` of last layer.
        use_conv: whether to use convolutional layer to replace the last layer, default to False.
        pool: parameters for the pooling layer, it should be a tuple, the first item is name of the pooling layer,
            the second item is dictionary of the initialization args. if None, will not replace the `layers[-2]`.
            default to `("avg", {"kernel_size": 7, "stride": 1})`.
        bias: the bias value when replacing the last layer. if False, the layer will not learn an additive bias,
            default to True.
        fc_name: the corresponding layer attribute of the last fully connected layer. Defaults to ``"fc"``.

    .. deprecated:: 0.6.0
        ``n_classes`` is deprecated, use ``num_classes`` instead.

    """

    @deprecated_arg("n_classes", since="0.6")
    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int = 1,
        dim: int = 2,
        in_channels: Optional[int] = None,
        use_conv: bool = False,
        pool: Optional[Tuple[str, Dict[str, Any]]] = ("avg", {"kernel_size": 7, "stride": 1}),
        bias: bool = True,
        n_classes: Optional[int] = None,
        fc_name: str = "fc",
    ):
        super().__init__()
        # in case the new num_classes is default but you still call deprecated n_classes
        if n_classes is not None and num_classes == 1:
            num_classes = n_classes
        layers = list(model.children())
        try:
            orig_fc = look_up_named_module(fc_name, model)
        except ValueError:
            orig_fc = layers[-1]
        in_channels_: int
        if in_channels is None:
            if not hasattr(orig_fc, "in_features"):
                raise ValueError("please specify the input channels of last layer with arg `in_channels`.")
            in_channels_ = orig_fc.in_features  # type: ignore
        else:
            in_channels_ = in_channels

        if pool is None:
            # remove the last layer or replace it with an identity
            if look_up_named_module(fc_name, model):  # assuming fc is the last layer
                self.features = set_named_module(model, fc_name, torch.nn.Identity())
            else:
                self.features = torch.nn.Sequential(*layers[:-1])
            self.pool = None
        else:
            # remove the last 2 layers
            self.features = torch.nn.Sequential(*layers[:-2])
            self.pool = get_pool_layer(name=pool, spatial_dims=dim)

        self.fc: Union[torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d]
        if use_conv:
            # add 1x1 conv (it behaves like a FC layer)
            self.fc = Conv[Conv.CONV, dim](in_channels=in_channels_, out_channels=num_classes, kernel_size=1, bias=bias)
        else:
            # replace the out_features of FC layer
            self.fc = torch.nn.Linear(in_features=in_channels_, out_features=num_classes, bias=bias)
        self.use_conv = use_conv
        self.dim = dim

    def forward(self, x):
        x = self.features(x)
        if isinstance(x, tuple):
            x = x[0]  # it might be a namedtuple such as torchvision.model.InceptionOutputs
        if self.pool is not None:
            x = self.pool(x)

        if not self.use_conv:
            x = torch.flatten(x, 1)
        else:  # user specified `use_conv` but the pooling layer removed the spatial dims
            while len(x.shape) < self.dim + 2:
                x = x[..., None]
        x = self.fc(x)

        return x

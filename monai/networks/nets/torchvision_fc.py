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

from typing import Any, Dict, Optional, Tuple

from monai.networks.nets import NetAdapter
from monai.utils import deprecated_arg, optional_import

models, _ = optional_import("torchvision.models")


__all__ = ["TorchVisionFCModel"]


class TorchVisionFCModel(NetAdapter):
    """
    Customize the fully connected layer of TorchVision model or replace it by convolutional layer.

    Args:
        model_name: name of any torchvision model with fully connected layer at the end.
            ``resnet18`` (default), ``resnet34``, ``resnet50``, ``resnet101``, ``resnet152``,
            ``resnext50_32x4d``, ``resnext101_32x8d``, ``wide_resnet50_2``, ``wide_resnet101_2``.
            model details: https://pytorch.org/vision/stable/models.html.
        num_classes: number of classes for the last classification layer. Default to 1.
        dim: number of supported spatial dimensions in the specified model, depends on the model implementation.
            default to 2 as most Torchvision models are for 2D image processing.
        in_channels: number of the input channels of last layer. if None, get it from `in_features` of last layer.
        use_conv: whether use convolutional layer to replace the last layer, default to False.
        pool: parameters for the pooling layer, it should be a tuple, the first item is name of the pooling layer,
            the second item is dictionary of the initialization args. if None, will not replace the `layers[-2]`.
            default to `("avg", {"kernel_size": 7, "stride": 1})`.
        bias: the bias value when replacing the last layer. if False, the layer will not learn an additive bias,
            default to True.
        pretrained: whether to use the imagenet pretrained weights. Default to False.
    """

    @deprecated_arg("n_classes", since="0.6")
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 1,
        dim: int = 2,
        in_channels: Optional[int] = None,
        use_conv: bool = False,
        pool: Optional[Tuple[str, Dict[str, Any]]] = ("avg", {"kernel_size": 7, "stride": 1}),
        bias: bool = True,
        pretrained: bool = False,
        n_classes: Optional[int] = None,
    ):
        # in case the new num_classes is default but you still call deprecated n_classes
        if n_classes is not None and num_classes == 1:
            num_classes = n_classes
        model = getattr(models, model_name)(pretrained=pretrained)
        # check if the model is compatible, should have a FC layer at the end
        if not str(list(model.children())[-1]).startswith("Linear"):
            raise ValueError(f"Model ['{model_name}'] does not have a Linear layer at the end.")

        super().__init__(
            model=model,
            num_classes=num_classes,
            dim=dim,
            in_channels=in_channels,
            use_conv=use_conv,
            pool=pool,
            bias=bias,
        )

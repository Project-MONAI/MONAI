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
"""
Defines factories for creating layers in generic, extensible, and dimensionally independent ways. A separate factory
object is created for each type of layer, and factory functions keyed to names are added to these objects. Whenever
a layer is requested the factory name and any necessary arguments are passed to the factory object. The return value
is typically a type but can be any callable producing a layer object.

The factory objects contain functions keyed to names converted to upper case, these names can be referred to as members
of the factory so that they can function as constant identifiers. eg. instance normalization is named `Norm.INSTANCE`.

For example, to get a transpose convolution layer the name is needed and then a dimension argument is provided which is
passed to the factory function:

.. code-block:: python

    dimension = 3
    name = Conv.CONVTRANS
    conv = Conv[name, dimension]

This allows the `dimension` value to be set in the constructor, for example so that the dimensionality of a network is
parameterizable. Not all factories require arguments after the name, the caller must be aware which are required.

Defining new factories involves creating the object then associating it with factory functions:

.. code-block:: python

    fact = LayerFactory()

    @fact.factory_function('test')
    def make_something(x, y):
        # do something with x and y to choose which layer type to return
        return SomeLayerType
    ...

    # request object from factory TEST with 1 and 2 as values for x and y
    layer = fact[fact.TEST, 1, 2]

Typically the caller of a factory would know what arguments to pass (ie. the dimensionality of the requested type) but
can be parameterized with the factory name and the arguments to pass to the created type at instantiation time:

.. code-block:: python

    def use_factory(fact_args):
        fact_name, type_args = split_args
        layer_type = fact[fact_name, 1, 2]
        return layer_type(**type_args)
    ...

    kw_args = {'arg0':0, 'arg1':True}
    layer = use_factory( (fact.TEST, kwargs) )
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch.nn as nn

from monai.networks.utils import has_nvfuser_instance_norm
from monai.utils import ComponentStore, look_up_option, optional_import

__all__ = ["LayerFactory", "Dropout", "Norm", "Act", "Conv", "Pool", "Pad", "RelPosEmbedding", "split_args"]


class LayerFactory(ComponentStore):
    """
    Factory object for creating layers, this uses given factory functions to actually produce the types or constructing
    callables. These functions are referred to by name and can be added at any time.
    """

    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)
        self.__doc__ = (
            f"Layer Factory '{name}': {description}\n".strip()
            + "\nPlease see :py:class:`monai.networks.layers.split_args` for additional args parsing."
            + "\n\nThe supported members are:"
        )

    def add_factory_callable(self, name: str, func: Callable, desc: str | None = None) -> None:
        """
        Add the factory function to this object under the given name, with optional description.
        """
        description: str = desc or func.__doc__ or ""
        self.add(name.upper(), description, func)
        # append name to the docstring
        assert self.__doc__ is not None
        self.__doc__ += f"{', ' if len(self.names)>1 else ' '}``{name}``"

    def add_factory_class(self, name: str, cls: type, desc: str | None = None) -> None:
        """
        Adds a factory function which returns the supplied class under the given name, with optional description.
        """
        self.add_factory_callable(name, lambda x=None: cls, desc)

    def factory_function(self, name: str) -> Callable:
        """
        Decorator for adding a factory function with the given name.
        """

        def _add(func: Callable) -> Callable:
            self.add_factory_callable(name, func)
            return func

        return _add

    def get_constructor(self, factory_name: str, *args) -> Any:
        """
        Get the constructor for the given factory name and arguments.

        Raises:
            TypeError: When ``factory_name`` is not a ``str``.

        """

        if not isinstance(factory_name, str):
            raise TypeError(f"factory_name must a str but is {type(factory_name).__name__}.")

        component = look_up_option(factory_name.upper(), self.components)

        return component.value(*args)

    def __getitem__(self, args) -> Any:
        """
        Get the given name or name/arguments pair. If `args` is a callable it is assumed to be the constructor
        itself and is returned, otherwise it should be the factory name or a pair containing the name and arguments.
        """

        # `args[0]` is actually a type or constructor
        if callable(args):
            return args

        # `args` is a factory name or a name with arguments
        if isinstance(args, str):
            name_obj, args = args, ()
        else:
            name_obj, *args = args

        return self.get_constructor(name_obj, *args)

    def __getattr__(self, key):
        """
        If `key` is a factory name, return it, otherwise behave as inherited. This allows referring to factory names
        as if they were constants, eg. `Fact.FOO` for a factory Fact with factory function foo.
        """

        if key in self.components:
            return key

        return super().__getattribute__(key)


def split_args(args):
    """
    Split arguments in a way to be suitable for using with the factory types. If `args` is a string it's interpreted as
    the type name.

    Args:
        args (str or a tuple of object name and kwarg dict): input arguments to be parsed.

    Raises:
        TypeError: When ``args`` type is not in ``Union[str, Tuple[Union[str, Callable], dict]]``.

    Examples::

        >>> act_type, args = split_args("PRELU")
        >>> monai.networks.layers.Act[act_type]
        <class 'torch.nn.modules.activation.PReLU'>

        >>> act_type, args = split_args(("PRELU", {"num_parameters": 1, "init": 0.25}))
        >>> monai.networks.layers.Act[act_type](**args)
        PReLU(num_parameters=1)

    """

    if isinstance(args, str):
        return args, {}
    name_obj, name_args = args

    if not (isinstance(name_obj, str) or callable(name_obj)) or not isinstance(name_args, dict):
        msg = "Layer specifiers must be single strings or pairs of the form (name/object-types, argument dict)"
        raise TypeError(msg)

    return name_obj, name_args


# Define factories for these layer types
Dropout = LayerFactory(name="Dropout layers", description="Factory for creating dropout layers.")
Norm = LayerFactory(name="Normalization layers", description="Factory for creating normalization layers.")
Act = LayerFactory(name="Activation layers", description="Factory for creating activation layers.")
Conv = LayerFactory(name="Convolution layers", description="Factory for creating convolution layers.")
Pool = LayerFactory(name="Pooling layers", description="Factory for creating pooling layers.")
Pad = LayerFactory(name="Padding layers", description="Factory for creating padding layers.")
RelPosEmbedding = LayerFactory(
    name="Relative positional embedding layers",
    description="Factory for creating relative positional embedding factory",
)


@Dropout.factory_function("dropout")
def dropout_factory(dim: int) -> type[nn.Dropout | nn.Dropout2d | nn.Dropout3d]:
    """
    Dropout layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the dropout layer

    Returns:
        Dropout[dim]d
    """
    types = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)
    return types[dim - 1]


Dropout.add_factory_class("alphadropout", nn.AlphaDropout)


@Norm.factory_function("instance")
def instance_factory(dim: int) -> type[nn.InstanceNorm1d | nn.InstanceNorm2d | nn.InstanceNorm3d]:
    """
    Instance normalization layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the instance normalization layer

    Returns:
        InstanceNorm[dim]d
    """
    types = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    return types[dim - 1]


@Norm.factory_function("batch")
def batch_factory(dim: int) -> type[nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d]:
    """
    Batch normalization layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the batch normalization layer

    Returns:
        BatchNorm[dim]d
    """
    types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    return types[dim - 1]


@Norm.factory_function("instance_nvfuser")
def instance_nvfuser_factory(dim):
    """
    `InstanceNorm3dNVFuser` is a faster version of InstanceNorm layer and implemented in `apex`.
    It only supports 3d tensors as the input. It also requires to use with CUDA and non-Windows OS.
    In this function, if the required library `apex.normalization.InstanceNorm3dNVFuser` does not exist,
    `nn.InstanceNorm3d` will be returned instead.
    This layer is based on a customized autograd function, which is not supported in TorchScript currently.
    Please switch to use `nn.InstanceNorm3d` if TorchScript is necessary.

    Please check the following link for more details about how to install `apex`:
    https://github.com/NVIDIA/apex#installation

    """

    if dim != 3:
        types = (nn.InstanceNorm1d, nn.InstanceNorm2d)
        warnings.warn(f"`InstanceNorm3dNVFuser` only supports 3d cases, use {types[dim - 1]} instead.")
        return types[dim - 1]

    if not has_nvfuser_instance_norm():
        warnings.warn(
            "`apex.normalization.InstanceNorm3dNVFuser` is not installed properly, use nn.InstanceNorm3d instead."
        )
        return nn.InstanceNorm3d
    return optional_import("apex.normalization", name="InstanceNorm3dNVFuser")[0]


Norm.add_factory_class("group", nn.GroupNorm)
Norm.add_factory_class("layer", nn.LayerNorm)
Norm.add_factory_class("localresponse", nn.LocalResponseNorm)
Norm.add_factory_class("syncbatch", nn.SyncBatchNorm)

Act.add_factory_class("elu", nn.modules.ELU)
Act.add_factory_class("relu", nn.modules.ReLU)
Act.add_factory_class("leakyrelu", nn.modules.LeakyReLU)
Act.add_factory_class("prelu", nn.modules.PReLU)
Act.add_factory_class("relu6", nn.modules.ReLU6)
Act.add_factory_class("selu", nn.modules.SELU)
Act.add_factory_class("celu", nn.modules.CELU)
Act.add_factory_class("gelu", nn.modules.GELU)
Act.add_factory_class("sigmoid", nn.modules.Sigmoid)
Act.add_factory_class("tanh", nn.modules.Tanh)
Act.add_factory_class("softmax", nn.modules.Softmax)
Act.add_factory_class("logsoftmax", nn.modules.LogSoftmax)


@Act.factory_function("swish")
def swish_factory():
    """
    Swish activation layer.

    Returns:
        Swish
    """
    from monai.networks.blocks.activation import Swish

    return Swish


@Act.factory_function("memswish")
def memswish_factory():
    """
    Memory efficient swish activation layer.

    Returns:
        MemoryEfficientSwish
    """
    from monai.networks.blocks.activation import MemoryEfficientSwish

    return MemoryEfficientSwish


@Act.factory_function("mish")
def mish_factory():
    """
    Mish activation layer.

    Returns:
        Mish
    """
    from monai.networks.blocks.activation import Mish

    return Mish


@Act.factory_function("geglu")
def geglu_factory():
    """
    GEGLU activation layer.

    Returns:
        GEGLU
    """
    from monai.networks.blocks.activation import GEGLU

    return GEGLU


@Conv.factory_function("conv")
def conv_factory(dim: int) -> type[nn.Conv1d | nn.Conv2d | nn.Conv3d]:
    """
    Convolutional layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the convolutional layer

    Returns:
        Conv[dim]d
    """
    types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    return types[dim - 1]


@Conv.factory_function("convtrans")
def convtrans_factory(dim: int) -> type[nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d]:
    """
    Transposed convolutional layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the transposed convolutional layer

    Returns:
        ConvTranspose[dim]d
    """
    types = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    return types[dim - 1]


@Pool.factory_function("max")
def maxpooling_factory(dim: int) -> type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d]:
    """
    Max pooling layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the max pooling layer

    Returns:
        MaxPool[dim]d
    """
    types = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)
    return types[dim - 1]


@Pool.factory_function("adaptivemax")
def adaptive_maxpooling_factory(dim: int) -> type[nn.AdaptiveMaxPool1d | nn.AdaptiveMaxPool2d | nn.AdaptiveMaxPool3d]:
    """
    Adaptive max pooling layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the adaptive max pooling layer

    Returns:
        AdaptiveMaxPool[dim]d
    """
    types = (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)
    return types[dim - 1]


@Pool.factory_function("avg")
def avgpooling_factory(dim: int) -> type[nn.AvgPool1d | nn.AvgPool2d | nn.AvgPool3d]:
    """
    Average pooling layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the average pooling layer

    Returns:
        AvgPool[dim]d
    """
    types = (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)
    return types[dim - 1]


@Pool.factory_function("adaptiveavg")
def adaptive_avgpooling_factory(dim: int) -> type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d]:
    """
    Adaptive average pooling layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the adaptive average pooling layer

    Returns:
        AdaptiveAvgPool[dim]d
    """
    types = (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)
    return types[dim - 1]


@Pad.factory_function("replicationpad")
def replication_pad_factory(dim: int) -> type[nn.ReplicationPad1d | nn.ReplicationPad2d | nn.ReplicationPad3d]:
    """
    Replication padding layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the replication padding layer

    Returns:
        ReplicationPad[dim]d
    """
    types = (nn.ReplicationPad1d, nn.ReplicationPad2d, nn.ReplicationPad3d)
    return types[dim - 1]


@Pad.factory_function("constantpad")
def constant_pad_factory(dim: int) -> type[nn.ConstantPad1d | nn.ConstantPad2d | nn.ConstantPad3d]:
    """
    Constant padding layers in 1,2,3 dimensions.

    Args:
        dim: desired dimension of the constant padding layer

    Returns:
        ConstantPad[dim]d
    """
    types = (nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d)
    return types[dim - 1]


@RelPosEmbedding.factory_function("decomposed")
def decomposed_rel_pos_embedding() -> type[nn.Module]:
    from monai.networks.blocks.rel_pos_embedding import DecomposedRelativePosEmbedding

    return DecomposedRelativePosEmbedding

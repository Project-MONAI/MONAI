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
"""
Defines factories for creating layers in generic, extensible, and dimensionally independent ways. A separate factory
object is created for each type of layer, and factory functions keyed to names are added to these objects. Whenever
a layer is requested the factory name and any necessary arguments are passed to the factory object. The return value
is typically a type but can be any callable producing a layer object. 

The factory objects contain functions keyed to names converted to upper case, these names can be referred to as members
of the factory so that they can function as constant identifiers. eg. instance normalisation is named `Norm.INSTANCE`.

For example, to get a transpose convolution layer the name is needed and then a dimension argument is provided which is 
passed to the factory function:

    dimension = 3
    name = Conv.CONVTRANS
    conv = Conv[name, dimension]
    
This allows the `dimension` value to be set in constructor for example so that the dimensionality of a network is 
parameterizable. Not all factories require arguments after the name, the caller must be aware which are required.

"""

from typing import Callable

import torch
import torch.nn as nn


class LayerFactory:
    """
    Factory object for creating layers, this uses given factory functions to actually produce the types or constructing
    callables. These functions are referred to by name and can be added at any time.
    """
    
    def __init__(self):
        self.factories = {}

    @property
    def names(self):
        """
        Produces all factory names.
        """
        
        return tuple(self.factories)

    def add_factory_callable(self, name, func):
        """
        Add the factory function to this object under the given name.
        """
        
        self.factories[name.upper()] = func

    def factory_function(self, name):
        """
        Decorator for adding a factory function with the given name.
        """
        
        def _add(func):
            self.add_factory_callable(name, func)
            return func

        return _add
    
    def get_constructor(self, factory_name, *args):
        """
        Get the constructor for the given factory name and arguments.
        """
        
        if not isinstance(factory_name, str):
            raise ValueError("Factories must be selected by name")
            
        fact = self.factories[factory_name.upper()]
        return fact(*args)

    def __getitem__(self, args):
        """
        Get the given name or name/arguments pair. If `args` is a callable it is assumed to be the constructor
        itself and is returned, otherwise it should be the factory name or a pair containing the name and arguments.
        """
        
        # `args` is actually a type or constructor
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
        
        if key in self.factories:
            return key

        return super().__getattr__(key)

    
def split_args(args):
    """
    Split arguments in a way to be suitable for using with the factory types. If `args` is a name it's interpreted 
    """
    
    if isinstance(args,str):
        return args,{}
    else:
        name_obj,args=args
        
        if not isinstance(name_obj,(str,Callable)) or not isinstance(args,dict):
            raise ValueError("Layer specifiers must be single strings or pairs of the form (name/object-types, argument dict)")
            
        return name_obj, args
    

### Define factories for these layer types
    
Dropout = LayerFactory()
Norm = LayerFactory()
Act = LayerFactory()
Conv = LayerFactory()


@Dropout.factory_function("dropout")
def get_dropout_type(dim):
    types = [nn.Dropout, nn.Dropout2d, nn.Dropout3d]
    return types[dim - 1]


@Norm.factory_function("instance")
def get_instance_type(dim):
    types = [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
    return types[dim - 1]


@Norm.factory_function("batch")
def get_batch_type(dim):
    types = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    return types[dim - 1]


Act.add_factory_callable("relu", lambda: nn.modules.ReLU)
Act.add_factory_callable("leakyrelu", lambda: nn.modules.LeakyReLU)
Act.add_factory_callable("prelu", lambda: nn.modules.PReLU)


@Conv.factory_function("conv")
def get_conv_type(dim):
    types = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
    return types[dim - 1]


@Conv.factory_function("convtrans")
def get_convtrans_type(dim):
    types = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    return types[dim - 1]


@Conv.factory_function("maxpool")
def get_maxpooling_type(dim, is_adaptive):
    if is_adaptive:
        types = [nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d]
    else:
        types = [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]
    return types[dim - 1]


@Conv.factory_function("avgpool")
def get_avgpooling_type(dim, is_adaptive):
    if is_adaptive:
        types = [nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]
    else:
        types = [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]
    return types[dim - 1]

### old factory functions


def get_conv_type(dim, is_transpose):
    if is_transpose:
        types = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    else:
        types = [nn.Conv1d, nn.Conv2d, nn.Conv3d]

    return types[dim - 1]


def get_dropout_type(dim):
    types = [nn.Dropout, nn.Dropout2d, nn.Dropout3d]
    return types[dim - 1]


def get_normalize_type(dim, is_instance):
    if is_instance:
        types = [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
    else:
        types = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]

    return types[dim - 1]


def get_maxpooling_type(dim, is_adaptive):
    if is_adaptive:
        types = [nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d]
    else:
        types = [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]
    return types[dim - 1]


def get_avgpooling_type(dim, is_adaptive):
    if is_adaptive:
        types = [nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]
    else:
        types = [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]
    return types[dim - 1]

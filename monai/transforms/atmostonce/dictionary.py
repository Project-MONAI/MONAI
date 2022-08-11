from typing import Mapping, Optional, Sequence, Union

import numpy as np

import torch

from monai.config import KeysCollection
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform
from monai.utils.enums import TransformBackends
from monai.utils.mapping_stack import MappingStack, MatrixFactory
from monai.utils.type_conversion import expand_scalar_to_tuple


def get_device_from_data(data):
    if isinstance(data, np.ndarray):
        return None
    elif isinstance(data, torch.Tensor):
        return data.device
    else:
        msg = "'data' must be one of numpy ndarray or torch Tensor but is {}"
        raise ValueError(msg.format(type(data)))


def get_backend_from_data(data):
    if isinstance(data, np.ndarray):
        return TransformBackends.NUMPY
    elif isinstance(data, torch.Tensor):
        return TransformBackends.TORCH
    else:
        msg = "'data' must be one of numpy ndarray or torch Tensor but is {}"
        raise ValueError(msg.format(type(data)))

# TODO: reconcile multiple definitions to one in utils
def expand_potential_tuple(keys, value):
    if not isinstance(value, (tuple, list)):
        return tuple(value for _ in keys)
    return value


class MappingStackTransformd(MapTransform, InvertibleTransform):

    def __init__(self,
                 keys: KeysCollection):
        super().__init__(self)
        self.keys = keys

    def __call__(self,
                 d: Mapping,
                 *args,
                 **kwargs):
        mappings = d.get("mappings", dict())
        rd = dict()
        for k in self.keys:
            data = d[k]
            dims = len(data.shape)-1
            device = get_device_from_data(data)
            backend = get_backend_from_data(data)
            v = mappings.get(k, MappingStack(MatrixFactory(dims, backend, device)))
            v.push(self.get_matrix(dims, backend, device, *args, **kwargs))
            mappings[k] = v
            rd[k] = data

        rd["mappings"] = mappings

        return rd

    def get_matrix(self, dims, backend, device, *args, **kwargs):
        msg = "get_matrix must be implemented in a subclass of MappingStackTransform"
        raise NotImplementedError(msg)


class RotateEulerd(MapTransform, InvertibleTransform):

    def __init__(self,
                 keys: KeysCollection,
                 euler_radians: Union[Sequence[float], float]):
        super().__init__(self)
        self.keys = keys
        self.euler_radians = expand_scalar_to_tuple(euler_radians, len(keys))

    def __call__(self, d: Mapping):
        mappings = d.get("mappings", dict())
        rd = dict()
        for k in self.keys:
            data = d[k]
            dims = len(data.shape)-1
            device = get_device_from_data(data)
            backend = get_backend_from_data(data)
            matrix_factory = MatrixFactory(dims, backend, device)
            v = mappings.get(k, MappingStack(matrix_factory))
            v.push(matrix_factory.rotate_euler(self.euler_radians))
            mappings[k] = v
            rd[k] = data

        rd["mappings"] = mappings

        return rd


class Translated(MapTransform, InvertibleTransform):

    def __init__(self,
                 keys: KeysCollection,
                 translate: Union[Sequence[float], float]):
        super().__init__(self)
        self.keys = keys
        self.translate = expand_scalar_to_tuple(translate, len(keys))

    def __call__(self, d: Mapping):
        mappings = d.get("mappings", dict())
        rd = dict()
        for k in self.keys:
            data = d[k]
            dims = len(data.shape)-1
            device = get_device_from_data(data)
            backend = get_backend_from_data(data)
            matrix_factory = MatrixFactory(dims, backend, device)
            v = mappings.get(k, MappingStack(matrix_factory))
            v.push(matrix_factory.translate(self.translate))
            mappings[k] = v
            rd[k] = data

        rd["mappings"] = mappings

        return rd


class Zoomd(MapTransform, InvertibleTransform):

    def __init__(self,
                 keys: KeysCollection,
                 scale: Union[Sequence[float], float]):
        super().__init__(self)
        self.keys = keys
        self.scale = expand_scalar_to_tuple(scale, len(keys))

    def __call__(self, d: Mapping):
        mappings = d.get("mappings", dict())
        rd = dict()
        for k in self.keys:
            data = d[k]
            dims = len(data.shape)-1
            device = get_device_from_data(data)
            backend = get_backend_from_data(data)
            matrix_factory = MatrixFactory(dims, backend, device)
            v = mappings.get(k, MappingStack(matrix_factory))
            v.push(matrix_factory.scale(self.scale))
            mappings[k] = v
            rd[k] = data

        rd["mappings"] = mappings

        return rd


# class RotateEulerd(MappingStackTransformd):
#
#     def __init__(self,
#                  keys: KeysCollection,
#                  euler_radians: Union[Sequence[float], float]):
#         super().__init__(keys)
#         self.euler_radians = euler_radians
#
#     def get_matrix(self, dims, backend, device, *args, **kwargs):
#         euler_radians = args[0] if len(args) > 0 else None
#         euler_radians = kwargs.get("euler_radians", None) if euler_radians is None
#         euler_radians = self.euler_radians if euler_radians is None else self.euler_radians
#         if euler_radians is None:
#             raise ValueError("'euler_radians' must be set during initialisation or passed in"
#                              "during __call__")
#         arg = euler_radians if self.euler_radians is None else euler_radians
#         return MatrixFactory(dims, backend, device).rotate_euler(arg)
#
#
# class ScaleEulerd(MappingStackTransformd):
#
#     def __init__(self,
#                  keys: KeysCollection,
#                  scale: Union[Sequence[float], float]):
#         super().__init__(keys)
#         self.scale = scale
#
#     def get_matrix(self, dims, backend, device, *args, **kwargs):
#         scale = args[0] if len(args) > 0 else None
#         scale = kwargs.get("scale", None) if scale is None
#         scale = self.scale if scale is None else self.euler_radians
#         if scale is None:
#             raise ValueError("'scale' must be set during initialisation or passed in"
#                              "during __call__")
#         arg = scale if self.scale is None else scale
#         return MatrixFactory(dims, backend, device).scale(arg)
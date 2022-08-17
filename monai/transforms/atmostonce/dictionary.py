from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

import torch

from monai.transforms.atmostonce.functional import rotate
from monai.utils import ensure_tuple, ensure_tuple_rep

from monai.config import KeysCollection, DtypeLike, SequenceStr
from monai.transforms.atmostonce.apply import apply
from monai.transforms.atmostonce.lazy_transform import LazyTransform
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform
from monai.utils.enums import TransformBackends, GridSampleMode, GridSamplePadMode
from monai.utils.mapping_stack import MatrixFactory, MetaMatrix
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
            v = None # mappings.get(k, MappingStack(MatrixFactory(dims, backend, device)))
            v.push(self.get_matrix(dims, backend, device, *args, **kwargs))
            mappings[k] = v
            rd[k] = data

        rd["mappings"] = mappings

        return rd

    def get_matrix(self, dims, backend, device, *args, **kwargs):
        msg = "get_matrix must be implemented in a subclass of MappingStackTransform"
        raise NotImplementedError(msg)


class Rotated(LazyTransform, MapTransform, InvertibleTransform):

    def __init__(self,
                 keys: KeysCollection,
                 angle: Union[Sequence[float], float],
                 keep_size: bool = True,
                 mode: Optional[SequenceStr] = GridSampleMode.BILINEAR,
                 padding_mode: Optional[SequenceStr] = GridSamplePadMode.BORDER,
                 align_corners: Optional[Union[Sequence[bool], bool]] = False,
                 dtype: Optional[Union[Sequence[Union[DtypeLike, torch.dtype]], DtypeLike, torch.dtype]] = np.float32,
                 allow_missing_keys: Optional[bool] = False,
                 lazy_evaluation: Optional[bool] = False
                 ):
        super().__init__(self)
        self.keys = keys
        self.angle = angle
        self.keep_size = keep_size
        self.modes = ensure_tuple_rep(mode, len(keys))
        self.padding_modes = ensure_tuple_rep(padding_mode, len(keys))
        self.align_corners = align_corners
        self.dtypes = ensure_tuple_rep(dtype, len(keys))
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, d: Mapping):
        rd = dict(d)
        if self.allow_missing_keys is True:
            keys_present = {k for k in self.keys if k in d}
        else:
            keys_present = self.keys

        for ik, k in enumerate(keys_present):
            img = d[k]

            img_t, transform, metadata = rotate(img, self.angle, self.keep_size,
                                                self.modes[ik], self.padding_modes[ik],
                                                self.align_corners, self.dtypes[ik])

            # TODO: candidate for refactoring into a LazyTransform method
            img_t.push_pending_transform(MetaMatrix(transform, metadata))
            if not self.lazy_evaluation:
                img_t = apply(img_t)

            rd[k] = img_t

        return rd

    def inverse(self, data: Any):
        raise NotImplementedError()


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
            v = None # mappings.get(k, MappingStack(matrix_factory))
            v.push(matrix_factory.translate(self.translate))
            mappings[k] = v
            rd[k] = data

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
            v = None # mappings.get(k, MappingStack(matrix_factory))
            v.push(matrix_factory.scale(self.scale))
            mappings[k] = v
            rd[k] = data

        rd["mappings"] = mappings

        return rd


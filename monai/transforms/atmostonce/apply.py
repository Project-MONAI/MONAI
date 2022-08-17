from typing import Dict, Hashable, Mapping, Optional, Sequence, Union

import itertools as it

import numpy as np

import torch

from monai.config import USE_COMPILED, DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data import MetaTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.array import Resample
from monai.transforms.transform import MapTransform
from monai.transforms.utils import create_grid
from monai.utils import GridSampleMode, GridSamplePadMode
from monai.utils.enums import TransformBackends
from monai.utils.misc import get_backend_from_data, get_device_from_data
from monai.utils.type_conversion import (convert_data_type, convert_to_dst_type,
                                         expand_scalar_to_tuple)
from monai.utils.mapping_stack import MatrixFactory

# TODO: This should move to a common place to be shared with dictionary
GridSampleModeSequence = Union[Sequence[Union[GridSampleMode, str]], GridSampleMode, str]
GridSamplePadModeSequence = Union[Sequence[Union[GridSamplePadMode, str]], GridSamplePadMode, str]
DtypeSequence = Union[Sequence[DtypeLike], DtypeLike]


# TODO: move to mapping_stack.py
def extents_from_shape(shape):
    extents = [[0, shape[i]] for i in range(1, len(shape))]

    extents = it.product(*extents)
    return list(np.asarray(e + (1,)) for e in extents)


# TODO: move to mapping_stack.py
def shape_from_extents(extents):
    aextents = np.asarray(extents)
    mins = aextents.min(axis=0)
    maxes = aextents.max(axis=0)
    return np.ceil(maxes - mins)[:-1].astype(int)


def apply(data: MetaTensor):
    pending = data.pending_transforms

    if len(pending) == 0:
        return data

    dim_count = len(data) - 1
    matrix_factory = MatrixFactory(dim_count,
                                   get_backend_from_data(data),
                                   get_device_from_data(data))

    # set up the identity matrix and metadata
    cumulative_matrix = matrix_factory.identity(dim_count)
    cumulative_extents = extents_from_shape(data.shape)

    # pre-translate origin to centre of image
    translate_to_centre = matrix_factory.translate(dim_count)
    cumulative_matrix = translate_to_centre @ cumulative_matrix
    cumulative_extents = [e @ translate_to_centre for e in cumulative_extents]

    for meta_matrix in pending:
        next_matrix = meta_matrix.matrix
        cumulative_matrix = next_matrix @ cumulative_matrix
        cumulative_extents = [e @ translate_to_centre for e in cumulative_extents]

    # TODO: figure out how to propagate extents properly
    # TODO: resampling strategy: augment resample or perform multiple stages if necessary
    # TODO: resampling strategy - antialiasing: can resample just be augmented?


    data.clear_pending_transforms()


class Apply(InvertibleTransform):

    def __init__(self):
        super().__init__()
        pass


class Applyd(MapTransform, InvertibleTransform):

    def __init__(self):
        super().__init__()
        pass

# class Applyd(MapTransform, InvertibleTransform):
#
#     def __init__(self,
#                  keys: KeysCollection,
#                  modes: GridSampleModeSequence,
#                  padding_modes: GridSamplePadModeSequence,
#                  normalized: bool = False,
#                  device: Optional[torch.device] = None,
#                  dtypes: Optional[DtypeSequence] = np.float32):
#         self.keys = keys
#         self.modes = modes
#         self.padding_modes = padding_modes
#         self.device = device
#         self.dtypes = dtypes
#         self.resamplers = dict()
#
#         if isinstance(dtypes, (list, tuple)):
#             if len(keys) != len(dtypes):
#                 raise ValueError("'keys' and 'dtypes' must be the same length if 'dtypes' is a sequence")
#
#             # create a resampler for each output data type
#             unique_resamplers = dict()
#             for d in dtypes:
#                 if d not in unique_resamplers:
#                     unique_resamplers[d] = Resample(norm_coords=not normalized, device=device, dtype=d)
#
#             # assign each named data input the appropriate resampler for that data type
#             for k, d in zip(keys, dtypes):
#                 if k not in self.resamplers:
#                     self.resamplers[k] = unique_resamplers[d]
#
#         else:
#             # share the resampler across all named data inputs
#             resampler = Resample(norm_coords=not normalized, device=device, dtype=dtypes)
#             for k in keys:
#                 self.resamplers[k] = resampler
#
#     def __call__(self,
#                  data: Mapping[Hashable, NdarrayOrTensor],
#                  allow_missing_keys: bool = False) -> Dict[Hashable, NdarrayOrTensor]:
#         d = dict(data)
#         mapping_stack = d["mappings"]
#         keys = d.keys()
#         for key_tuple in self.key_iterator(d,
#                                            expand_scalar_to_tuple(self.modes, len(keys)),
#                                            expand_scalar_to_tuple(self.padding_modes, len(keys)),
#                                            expand_scalar_to_tuple(self.dtypes, len(keys))):
#             key, mode, padding_mode, dtype = key_tuple
#             affine = mapping_stack[key].transform()
#             data = d[key]
#             spatial_size = data.shape[1:]
#             grid = create_grid(spatial_size, device=self.device, backend="torch", dtype=dtype)
#             _device = grid.device
#
#             _b = TransformBackends.TORCH if isinstance(grid, torch.Tensor) else TransformBackends.NUMPY
#
#             grid, *_ = convert_data_type(grid, torch.Tensor, device=_device, dtype=grid.dtype)
#             affine, *_ = convert_to_dst_type(affine, grid)
#             d[key] = self.resamplers[key](data, grid=grid, mode=mode, padding_mode=padding_mode)
#
#         return d

from typing import Sequence, Union

import itertools as it

import numpy as np

import torch
from monai.transforms import Affine

from monai.config import DtypeLike
from monai.data import MetaTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform
from monai.utils import GridSampleMode, GridSamplePadMode
from monai.utils.misc import get_backend_from_data, get_device_from_data
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
def shape_from_extents(
        src_shape: Sequence,
        extents: Union[Sequence[np.ndarray], Sequence[torch.Tensor], np.ndarray, torch.Tensor]
):
    if isinstance(extents, (list, tuple)):
        if isinstance(extents[0], np.ndarray):
            aextents = np.asarray(extents)
            aextents = torch.from_numpy(aextents)
        else:
            aextents = torch.stack(extents)
    else:
        if isinstance(extents, np.ndarray):
            aextents = torch.from_numpy(extents)
        else:
            aextents = extents

    mins = aextents.min(axis=0)[0]
    maxes = aextents.max(axis=0)[0]
    values = torch.ceil(maxes - mins).type(torch.IntTensor)[:-1]
    return torch.cat((torch.as_tensor([src_shape[0]]), values))

    # return [src_shape[0]] + np.ceil(maxes - mins)[:-1].astype(int).tolist()



def apply(data: MetaTensor):
    pending = data.pending_transforms

    if len(pending) == 0:
        return data

    dim_count = len(data.shape) - 1
    matrix_factory = MatrixFactory(dim_count,
                                   get_backend_from_data(data),
                                   get_device_from_data(data))

    # set up the identity matrix and metadata
    cumulative_matrix = matrix_factory.identity()
    cumulative_extents = extents_from_shape(data.shape)

    # pre-translate origin to centre of image
    translate_to_centre = matrix_factory.translate([d / 2 for d in data.shape[1:]])
    cumulative_matrix = translate_to_centre @ cumulative_matrix
    cumulative_extents = [e @ translate_to_centre.matrix.matrix for e in cumulative_extents]

    for meta_matrix in pending:
        next_matrix = meta_matrix.matrix
        cumulative_matrix = next_matrix @ cumulative_matrix
        cumulative_extents = [e @ translate_to_centre.matrix.matrix for e in cumulative_extents]

    # TODO: figure out how to propagate extents properly
    # TODO: resampling strategy: augment resample or perform multiple stages if necessary
    # TODO: resampling strategy - antialiasing: can resample just be augmented?
    # r = Resample()
    a = Affine(affine=cumulative_matrix.matrix.matrix,
               padding_mode=cur_padding_mode,
               spatial_size=cur_spatial_size)
    data.clear_pending_transforms()


class Apply(InvertibleTransform):

    def __init__(self):
        super().__init__()


class Applyd(MapTransform, InvertibleTransform):

    def __init__(self):
        super().__init__()

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

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

import torch

from monai.transforms.atmostonce.array import Rotate, Resize, Spacing, Zoom, CropPad
from monai.transforms.atmostonce.utility import ILazyTransform, IRandomizableTransform, IMultiSampleTransform
from monai.utils import ensure_tuple_rep

from monai.config import KeysCollection, DtypeLike, SequenceStr
from monai.transforms.atmostonce.lazy_transform import LazyTransform
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils.enums import TransformBackends, GridSampleMode, GridSamplePadMode, InterpolateMode, NumpyPadMode, \
    PytorchPadMode
from monai.utils.mapping_stack import MatrixFactory
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


def keys_to_process(
        keys: Sequence[str],
        dictionary: dict,
        allow_missing_keys: bool,
):
    if allow_missing_keys is True:
        return {k for k in keys if k in dictionary}
    return keys


# class MappingStackTransformd(MapTransform, InvertibleTransform):
#
#     def __init__(self,
#                  keys: KeysCollection):
#         super().__init__(self)
#         self.keys = keys
#
#     def __call__(self,
#                  d: Mapping,
#                  *args,
#                  **kwargs):
#         mappings = d.get("mappings", dict())
#         rd = dict()
#         for k in self.keys:
#             data = d[k]
#             dims = len(data.shape)-1
#             device = get_device_from_data(data)
#             backend = get_backend_from_data(data)
#             v = None # mappings.get(k, MappingStack(MatrixFactory(dims, backend, device)))
#             v.push(self.get_matrix(dims, backend, device, *args, **kwargs))
#             mappings[k] = v
#             rd[k] = data
#
#         rd["mappings"] = mappings
#
#         return rd
#
#     def get_matrix(self, dims, backend, device, *args, **kwargs):
#         msg = "get_matrix must be implemented in a subclass of MappingStackTransform"
#         raise NotImplementedError(msg)


class Spacingd(LazyTransform, MapTransform, InvertibleTransform):

    def __init__(self,
                 keys: KeysCollection,
                 pixdim: Union[Sequence[float], float, np.ndarray],
                 src_pixdim: Optional[Union[Sequence[float], float, np.ndarray]],
                 diagonal: Optional[bool] = False,
                 mode: Optional[Union[GridSampleMode, str]] = GridSampleMode.BILINEAR,
                 padding_mode: Optional[Union[GridSamplePadMode, str]] = GridSamplePadMode.BORDER,
                 align_corners: Optional[bool] = False,
                 dtype: Optional[DtypeLike] = np.float64,
                 allow_missing_keys: Optional[bool] = False,
                 lazy_evaluation: Optional[bool] = False
                 ):
        LazyTransform.__init__(self, lazy_evaluation)
        MapTransform.__init__(self)
        InvertibleTransform.__init__(self)
        self.keys = keys
        self.pixdim = pixdim
        self.src_pixdim = src_pixdim
        self.diagonal = diagonal
        self.modes = ensure_tuple_rep(mode)
        self.padding_modes = ensure_tuple_rep(padding_mode)
        self.align_corners = align_corners
        self.dtypes = ensure_tuple_rep(dtype)
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, d: Mapping):
        rd = dict(d)
        if self.allow_missing_keys is True:
            keys_present = {k for k in self.keys if k in d}
        else:
            keys_present = self.keys

        for ik, k in enumerate(keys_present):
            tx = Spacing(self.pixdim, self.src_pixdim, self.diagonal,
                         self.modes[ik], self.padding_modes[ik],
                         self.align_corners, self.dtypes[ik])

            rd[k] = tx(d[k])

        return rd


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
            tx = Rotate(self.angle, self.keep_size,
                        self.modes[ik], self.padding_modes[ik],
                        self.align_corners, self.dtypes[ik])
            rd[k] = tx(d[k])

        return rd

    def inverse(self, data: Any):
        raise NotImplementedError()


class Resized(LazyTransform, MapTransform, InvertibleTransform):

    def __init__(self,
                 keys: KeysCollection,
                 spatial_size: Union[Sequence[int], int],
                 size_mode: Optional[str] = "all",
                 mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
                 align_corners: Optional[bool] = False,
                 anti_aliasing: Optional[bool] = False,
                 anti_aliasing_sigma: Optional[Union[Sequence[float], float, None]] = None,
                 dtype: Optional[Union[DtypeLike, torch.dtype]] = np.float32,
                 lazy_evaluation: Optional[bool] = False
                 ):
        LazyTransform.__init__(self, lazy_evaluation)
        self.keys = keys
        self.spatial_size = spatial_size
        self.size_mode = size_mode
        self.modes = ensure_tuple_rep(mode),
        self.align_corners = align_corners
        self.anti_aliasing = anti_aliasing
        self.anti_aliasing_sigma = anti_aliasing_sigma
        self.dtype = dtype

    def __call__(self, d: Mapping):
        rd = dict(d)
        if self.allow_missing_keys is True:
            keys_present = {k for k in self.keys if k in d}
        else:
            keys_present = self.keys

        for ik, k in enumerate(keys_present):
            tx = Resize(self.spatial_size, self.size_mode, self.modes[ik], self.align_corners,
                        self.anti_aliasing, self.anti_aliasing_sigma, self.dtype)
            rd[k] = tx(d[k])

        return rd

    def inverse(self, data: Any):
        raise NotImplementedError()


class Zoomd(LazyTransform, MapTransform, InvertibleTransform):

    def __init__(self,
                 keys: KeysCollection,
                 zoom: Union[Sequence[float], float],
                 mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.AREA,
                 padding_mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = NumpyPadMode.EDGE,
                 align_corners: Optional[bool] = None,
                 keep_size: Optional[bool] = True,
                 dtype: Optional[Union[DtypeLike, torch.dtype]] = np.float32,
                 lazy_evaluation: Optional[bool] = False,
                 **kwargs
                 ):
        LazyTransform.__init__(self, lazy_evaluation)
        self.keys = keys
        self.zoom = zoom
        self.modes = ensure_tuple_rep(mode)
        self.padding_modes = ensure_tuple_rep(padding_mode)
        self.align_corners = align_corners
        self.keep_size = keep_size
        self.dtype = dtype

    def __call__(self, d: Mapping):
        rd = dict(d)
        if self.allow_missing_keys is True:
            keys_present = {k for k in self.keys if k in d}
        else:
            keys_present = self.keys

        for ik, k in enumerate(keys_present):
            tx = Zoom(self.zoom, self.modes[ik], self.padding_modes[k], self.align_corners,
                      self.keep_size, self.dtype)
            rd[k] = tx(d[k])

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


class CropPadd(MapTransform, InvertibleTransform, ILazyTransform):

    def __init__(
        self,
        keys: KeysCollection,
        slices: Optional[Sequence[slice]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = GridSamplePadMode.BORDER,
        lazy_evaluation: Optional[bool] = True
    ):
        self.keys = keys
        self.slices = slices
        self.padding_modes = padding_mode
        self.lazy_evaluation = lazy_evaluation


    def __call__(
        self,
        d: dict
    ):
        keys = keys_to_process(self.keys, d, self.allow_missing_keys)

        rd = dict(d)
        for ik, k in enumerate(keys):
            tx = CropPad(slices=self.slices,
                         padding_mode=self.padding_modes,
                         lazy_evaluation=self.lazy_evaluation)

            rd[k] = tx(d[k])

        return rd


class RandomCropPadd(MapTransform, InvertibleTransform, RandomizableTransform, ILazyTransform):

    def __init__(
            self,
            keys: KeysCollection,
            sizes: Union[Sequence[int], int],
            prob: Optional[float] = 0.1,
            padding_mode: Optional[Union[GridSamplePadMode, str]] = GridSamplePadMode.BORDER,
            allow_missing_keys: bool=False,
            lazy_evaluation: Optional[bool] = True
    ):
        RandomizableTransform.__init__(self, prob)
        self.keys = keys
        self.sizes = sizes
        self.padding_mode = padding_mode
        self.offsets = None
        self.allow_missing_keys = allow_missing_keys

        self.op = CropPad(None, padding_mode)

    def randomize(
            self,
            img: torch.Tensor,
    ):
        super().randomize(None)
        if self._do_transform:
            img_shape = img.shape[1:]
            if isinstance(self.sizes, int):
                crop_shape = tuple(self.sizes for _ in range(len(img_shape)))
            else:
                crop_shape = self.sizes

            valid_ranges = tuple(i - c for i, c in zip(img_shape, crop_shape))
            self.offsets = tuple(self.R.randint(0, r+1) if r > 0 else r for r in valid_ranges)

    def __call__(
            self,
            d: dict,
            randomize: Optional[bool] = True
    ):
        keys = keys_to_process(self.keys, d, self.allow_missing_keys)

        img = d[keys[0]]
        img_shape = img.shape[:1]

        if randomize:
            self.randomize(img)

        if self._do_transform:
            offsets_ = self.offsets
        else:
            # center crop if this sample isn't random
            offsets_ = tuple((i - s) // 2 for i, s in zip(img_shape, self.sizes))

        slices = tuple(slice(o, o + s) for o, s in zip(offsets_, self.sizes))

        rd = dict(d)
        for k in keys:
            rd[k] = self.op(img, slices=slices)

        return rd

    @property
    def lazy_evaluation(self):
        return self.op.lazy_evaluation


class RandomCropPadMultiSampled(
    InvertibleTransform, IRandomizableTransform, ILazyTransform, IMultiSampleTransform
):

    def __init__(
            self,
            keys: Sequence[str],
            sizes: Union[Sequence[int], int],
            sample_count: int,
            padding_mode: Optional[Union[GridSamplePadMode, str]] = GridSamplePadMode.BORDER,
            lazy_evaluation: Optional[bool] = True
    ):
        self.sample_count = sample_count
        self.op = RandomCropPadd(keys, sizes, 1.0, padding_mode, lazy_evaluation)

    def __call__(
            self,
            d: dict,
            randomize: Optional[bool] = True
    ):
        for i in range(self.sample_count):
            yield self.op(d, randomize)

    def inverse(
            self,
            data: dict
    ):
        raise NotImplementedError()

    def set_random_state(self, seed=None, state=None):
        self.op.set_random_state(seed, state)

    @property
    def lazy_evaluation(self):
        return self.op.lazy_evaluation
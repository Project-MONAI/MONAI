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
A collection of dictionary-based wrappers around the "vanilla" transforms for spatial operations
defined in :py:class:`monai.transforms.spatial.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from copy import deepcopy
from typing import Dict, Hashable, Mapping, Optional, Sequence, Union, List
import numpy as np
import torch
from enum import Enum
import random
import time

from monai.data.utils import orientation_ras_lps
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data import box_utils
from monai.transforms.box.array import BoxZoom, BoxClipToImage, BoxConvertMode, BoxConvertToStandard, BoxFlip, BoxAffine, BoxMaskToBox, BoxToBoxMask, BoxSpatialCropPad, BoxCropForeground
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Randomizable, RandomizableTransform
from monai.transforms.croppad.array import (
    RandCropByPosNegLabel,
    SpatialCrop,
    Pad,
)
from monai.transforms.spatial.array import (
    RandZoom,
    RandFlip
    )
from monai.transforms.utils import (
    generate_pos_neg_label_crop_centers,
)
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.utils.module import optional_import
from monai.utils.enums import PostFix, TraceKeys
from monai.utils import (
    ensure_tuple_rep,
    fall_back_tuple,
    NumpyPadMode,
    PytorchPadMode,
    InterpolateMode,
)
from monai.utils import ImageMetaKey as Key

nib, _ = optional_import("nibabel")

__all__ = [
    "BoxConvertToStandardd",
    "BoxConvertToStandardD",
    "BoxConvertToStandardDict",
    "BoxConvertModed",
    "BoxConvertModeD",
    "BoxConvertModeDict",
    "BoxClipToImaged",
    "BoxClipToImageD",
    "BoxClipToImageDict",
    "BoxFlipd",
    "BoxFlipD",
    "BoxFlipDict",
    "BoxToImageCoordinated",
    "BoxToImageCoordinateD",
    "BoxToImageCoordinateDict",
    "BoxMaskToBoxd",
    "BoxMaskToBoxD",
    "BoxMaskToBoxDict",
    "BoxToBoxMaskd",
    "BoxToBoxMaskD",
    "BoxToBoxMaskDict",
    "BoxCropForegroundd",
    "BoxCropForegroundD",
    "BoxCropForegroundDict",
    "BoxRandCropForegroundd",
    "BoxRandCropForegroundD",
    "BoxRandCropForegroundDict",
    "BoxRandZoomd",
    "BoxRandZoomD",
    "BoxRandZoomDict",
    "BoxRandFlipd",
    "BoxRandFlipD",
    "BoxRandFlipDict",
]

DEFAULT_POST_FIX = PostFix.meta()
InterpolateModeSequence = Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str]
PadModeSequence = Union[Sequence[Union[NumpyPadMode, PytorchPadMode, str]], NumpyPadMode, PytorchPadMode, str]

class BoxConvertToStandardd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.BoxFlip`.

    Args:
        box_keys: Keys to pick data for transformation.
        box_mode: String, if not given, we assume box mode is standard mode
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        box_mode: str,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        self.converter = BoxConvertToStandard(mode=box_mode)
        self.inverse_converter = BoxConvertMode(mode1=None, mode2=box_mode)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        key_id = 0
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.converter(d[key])
            key_id += 1
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        key_id = 0
        for key in self.key_iterator(d):
            _ = self.get_most_recent_transform(d, key)
            # Inverse is same as forward
            d[key] = self.inverse_converter(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
            key_id += 1
        return d


class BoxConvertModed(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.BoxFlip`.

    Args:
        box_keys: Keys to pick data for transformation.
        box_mode: String, if not given, we assume box mode is standard mode
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        box_mode: str,
        tgt_mode: str,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        self.converter = BoxConvertMode(mode1=box_mode, mode2=tgt_mode)
        self.inverse_converter = BoxConvertMode(mode1=tgt_mode, mode2=box_mode)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        key_id = 0
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.converter(d[key])
            key_id += 1
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        key_id = 0
        for key in self.key_iterator(d):
            _ = self.get_most_recent_transform(d, key)
            # Inverse is same as forward
            d[key] = self.inverse_converter(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
            key_id += 1
        return d


class BoxToImageCoordinated(MapTransform, InvertibleTransform):
    """
    Dictionary-based transfrom that converts box in world coordinate to image coordinate.

    Args:
        box_keys: Keys to pick data for transformation.
        image_key: Key for the images that boxes belong to
        box_mode: String, if not given, we assume box mode is standard mode
        remove_empty: whether to remove the boxes that are actually empty
        allow_missing_keys: don't raise exception if key is missing.
        image_meta_key: explicitly indicate the key of the corresponding meta data dictionary.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the meta data is a dictionary object which contains: filename, affine, original_shape, etc.
            it is a string, map to the `image_key`.
            if None, will try to construct meta_keys by `image_key_{meta_key_postfix}`.
        image_meta_key_postfix: if image_meta_keys=None, use `image_key_{postfix}` to fetch the meta data according
            to the key data, default is `meta_dict`, the meta data is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        affine_lps_to_ras: default False. if 1) the image is read by ITKReader, 
            2) the ITKReader has affine_lps_to_ras=True, and 3) the box is in world coordinate, 
            then set affine_lps_to_ras=True
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        image_key: str,
        box_mode: str = None,
        allow_missing_keys: bool = False,
        image_meta_key: str = None,
        image_meta_key_postfix: str = DEFAULT_POST_FIX,
        affine_lps_to_ras=False
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        self.image_meta_key = image_meta_key or f"{image_key}_{image_meta_key_postfix}"

        if (box_mode is not None) and (box_mode not in ["xxyy","xxyyzz","xyxy","xyzxyz","ccwh", "cccwhd"]):
            raise ValueError(
                f"We support only box mode in [xxyy,xxyyzz,xyxy,xyzxyz,ccwh, cccwhd], bt we got {box_mode}."
            )
        self.converter_to_image_coordinate = BoxAffine(mode=box_mode, invert_affine=True)
        self.converter_to_physical_coordinate = BoxAffine(mode=None, invert_affine=False)
        self.affine_lps_to_ras = affine_lps_to_ras

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        meta_key = self.image_meta_key
        for key in self.key_iterator(d): 
            # create metadata if necessary
            if meta_key not in d:
                d[meta_key] = {"affine": None}
            meta_data = d[meta_key]       
            affine = meta_data["affine"] # RAS affine
            if self.affine_lps_to_ras:
                affine = orientation_ras_lps(affine)
            self.push_transform(d, key, extra_info={"affine":affine})
            d[key] = self.converter_to_image_coordinate(d[key], affine=affine)
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        meta_key = self.image_meta_key
        for key in self.key_iterator(d): 
            transform = self.get_most_recent_transform(d, key)
            affine = transform['extra_info']['affine']
            d[key] = self.converter_to_physical_coordinate(d[key], affine=affine)
            self.pop_transform(d, key)
        return d

class BoxClipToImaged(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.BoxFlip`.

    Args:
        box_keys: Keys to pick data for transformation.
        image_key: Key for the images that boxes belong to
        box_mode: String, if not given, we assume box mode is standard mode
        remove_empty: whether to remove the boxes that are actually empty
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        label_keys: KeysCollection,
        image_key: str,
        box_mode: str = None,
        remove_empty: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)

        if (box_mode is not None) and (box_mode not in box_utils.STANDARD_MODE):
            raise ValueError(
                "Currently we support only standard box_mode."
                "Please apply BoxConvertToStandardd first and then set box_mode=None."
            )
        if len(box_keys)>1:
            raise ValueError("We support only one box_key. len(box_keys) should be 1.")
        self.label_keys = label_keys
        self.image_key = image_key
        self.box_mode = box_mode
        self.remove_empty = remove_empty

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        image_size = d[self.image_key].shape[1:]
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key], keep = box_utils.box_clip_to_image(d[key], image_size, self.box_mode, self.remove_empty)
        for label_key in self.label_keys:
            d[label_key] = d[label_key][keep]
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key, label_key in self.key_iterator(d, self.label_keys):
            self.pop_transform(d, key, label_key)
        return d


class BoxFlipd(MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip`.
    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    Args:
        image_keys: Image keys to pick data for transformation.
        box_keys: Box keys to pick data for transformation.
        ref_image_keys: reference image where the boxes lays on, default equal to image_keys
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandFlip.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        ref_image_keys: KeysCollection = None,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, image_keys, allow_missing_keys)
        self.flipper = Flip(spatial_axis=spatial_axis)
        self.box_flipper = BoxFlip(spatial_axis=spatial_axis, mode=None)
        self.box_keys = box_keys
        if ref_image_keys==None:
            ref_image_keys = image_keys
        self.ref_image_keys = ensure_tuple_rep(ref_image_keys, len(self.box_keys))

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            d[key] = self.flipper(d[key])
            self.push_transform(d, key)
                
        for box_key,ref_image_key in zip(self.box_keys,self.ref_image_keys): 
            image_size = d[ref_image_key].shape[1:]            
            d[box_key] = self.box_flipper(d[box_key], image_size)
            self.push_transform(d, box_key, extra_info={"image_size": image_size})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            d[key] = self.flipper(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        for box_key in self.box_keys:
            transform = self.get_most_recent_transform(d, box_key)
            # Create inverse transform
            image_size = np.array(transform[TraceKeys.EXTRA_INFO]["image_size"])
            d[box_key] = self.box_flipper(d[box_key], image_size)
            # Remove the applied transform
            self.pop_transform(d, box_key)
        return d

class BoxRandFlipd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip`.
    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    Args:
        image_keys: Image keys to pick data for transformation.
        box_keys: Box keys to pick data for transformation.
        ref_image_keys: reference image where the boxes lays on, default equal to image_keys
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandFlip.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        ref_image_keys: KeysCollection = None,
        prob: float = 0.1,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, image_keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.flipper = RandFlip(prob=1.0, spatial_axis=spatial_axis)
        self.box_flipper = BoxFlip(spatial_axis=spatial_axis, mode=None)
        self.box_keys = box_keys
        if ref_image_keys==None:
            ref_image_keys = image_keys
        self.ref_image_keys = ensure_tuple_rep(ref_image_keys, len(self.box_keys))
        

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandFlipd":
        super().set_random_state(seed, state)
        self.flipper.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = self.flipper(d[key], randomize=False)
            self.push_transform(d, key)
                
        for box_key,ref_image_key in zip(self.box_keys,self.ref_image_keys): 
            image_size = d[ref_image_key].shape[1:]            
            if self._do_transform:                               
                d[box_key] = self.box_flipper(d[box_key], image_size)
            self.push_transform(d, box_key, extra_info={"image_size": image_size})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # Inverse is same as forward
                d[key] = self.flipper(d[key], randomize=False)
            # Remove the applied transform
            self.pop_transform(d, key)

        for box_key in self.box_keys:
            transform = self.get_most_recent_transform(d, box_key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # Create inverse transform
                image_size = np.array(transform[TraceKeys.EXTRA_INFO]["image_size"])
                d[box_key] = self.box_flipper(d[box_key], image_size)
            # Remove the applied transform
            self.pop_transform(d, box_key)
        return d



class BoxRandSwapAxesd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip`.
    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    Args:
        image_keys: Image keys to pick data for transformation.
        box_keys: Box keys to pick data for transformation.
        ref_image_keys: reference image where the boxes lays on, default equal to image_keys
        prob: Probability of flipping.
        spatial_axis: Spatial axes that will be transposed
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandFlip.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        prob: float = 0.1,
        spatial_axis_0: int = 0,
        spatial_axis_1: int = 1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, image_keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.box_keys = box_keys
        if spatial_axis_0 == spatial_axis_1:
            raise ValueError("spatial_axis_0 and spatial_axis_1 should be different.")
        self.spatial_axis_0 = spatial_axis_0
        self.spatial_axis_1 = spatial_axis_1
        

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        first_key: Union[Hashable, List] = self.first_key(d)
        spatial_dims = len(d[first_key].shape)-1
        
        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = d[key].swapaxes(self.spatial_axis_0+1,self.spatial_axis_1+1)
            self.push_transform(d, key, extra_info={"swapaxes": [self.spatial_axis_0,self.spatial_axis_1]})
                
        for box_key in self.box_keys:            
            if self._do_transform:   
                if d[box_key].shape[0]>0:                          
                    d[box_key][:,[self.spatial_axis_0,self.spatial_axis_1]] = d[box_key][:,[self.spatial_axis_1,self.spatial_axis_0]]
                    d[box_key][:,[self.spatial_axis_0+spatial_dims,self.spatial_axis_1+spatial_dims]] = d[box_key][:,[self.spatial_axis_1+spatial_dims,self.spatial_axis_0+spatial_dims]]
            self.push_transform(d, box_key, extra_info={"swapaxes0": [self.spatial_axis_0,self.spatial_axis_1], "swapaxes1": [self.spatial_axis_0+spatial_dims,self.spatial_axis_1+spatial_dims]})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # Inverse is same as forward
                transposed_spatial_axis = transform[TraceKeys.EXTRA_INFO]["swapaxes"]
                d[key] = d[key].swapaxes(transposed_spatial_axis[0],transposed_spatial_axis[1])
            # Remove the applied transform
            self.pop_transform(d, key)

        for box_key in self.box_keys:
            transform = self.get_most_recent_transform(d, box_key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # Create inverse transform
                if d[box_key].shape[0]>0:
                    transposed_spatial_axis = transform[TraceKeys.EXTRA_INFO]["swapaxes0"]
                    d[box_key][:,[transposed_spatial_axis[0],transposed_spatial_axis[1]]] = d[box_key][:,[transposed_spatial_axis[1],transposed_spatial_axis[0]]]
                    transposed_spatial_axis = transform[TraceKeys.EXTRA_INFO]["swapaxes1"]
                    d[box_key][:,[transposed_spatial_axis[0],transposed_spatial_axis[1]]] = d[box_key][:,[transposed_spatial_axis[1],transposed_spatial_axis[0]]]
            # Remove the applied transform
            self.pop_transform(d, box_key)
        return d

class BoxToBoxMaskd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Flip`.
    Convert box to CxMxNxP mask image, which has the same size with the input image MxNxP.
    The channel number equals to the number of boxes

    Args:
        keys: Keys to pick data for transformation.
        min_fg_label: min foreground box label.
        allow_missing_keys: don't raise exception if key is missing.
        ellipse_mask: bool.
            If True, it assumes the object shape is close to ellipse or ellipsoid.
            If False, it assumes the object shape is close to rectangle or cube and well occupies the bounding box.
            If the users are going to apply random rotation as data augmentation, we suggest setting ellipse_mask=True
            See also https://openaccess.thecvf.com/content/ICCV2021/papers/Kalra_Towards_Rotation_Invariance_in_Object_Detection_ICCV_2021_paper.pdf
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        label_keys: KeysCollection,
        box_mask_keys: KeysCollection,
        image_key: str,
        min_fg_label: int = 0,
        box_mode: str = None,
        ellipse_mask: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        if (box_mode is not None) and (box_mode not in box_utils.STANDARD_MODE):
            raise ValueError(
                "Currently we support only standard box_mode."
                "Please apply BoxConvertToStandardd first and then set box_mode=None."
            )
        self.image_key = image_key
        if len(label_keys) != len(self.keys) or len(box_mask_keys) != len(self.keys):
            raise ValueError("Please make sure len(label_keys)==len(box_keys)==len(box_mask_keys)!")
        self.label_keys = label_keys
        self.box_mask_keys = box_mask_keys
        self.bg_label = min_fg_label - 1
        self.converter = BoxToBoxMask(mode=box_mode, bg_label=self.bg_label, ellipse_mask=ellipse_mask)


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        image_size = list(d[self.image_key].shape[1:])

        key_id = 0
        for key, label_key, box_mask_key in self.key_iterator(d, self.label_keys,self.box_mask_keys):
            # self.push_transform(d, key)
            # make the mask non-negative
            d[box_mask_key] = self.converter(d[key], image_size, d[label_key]) - self.bg_label
            key_id += 1
        return d


class BoxMaskToBoxd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Flip`.
    Convert mask to box

    Args:
        keys: Keys to pick data for transformation.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        min_fg_label: min foreground box label.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        label_keys: KeysCollection,
        box_mask_keys: KeysCollection,
        image_key: str,
        min_fg_label: int = 0,
        box_mode: str = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        if (box_mode is not None) and (box_mode not in box_utils.STANDARD_MODE):
            raise ValueError(
                "Currently we support only standard box_mode."
                "Please apply BoxConvertToStandardd first and then set box_mode=None."
            )
        self.image_key = image_key
        if len(label_keys) != len(self.keys) or len(box_mask_keys) != len(self.keys):
            raise ValueError("Please make sure len(label_keys)==len(box_keys)==len(box_mask_keys)!")
        self.label_keys = label_keys
        self.box_mask_keys = box_mask_keys
        self.bg_label = min_fg_label - 1
        self.converter = BoxMaskToBox(mode=box_mode, bg_label=self.bg_label)


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        image_size = d[self.image_key].shape[1:]

        key_id = 0
        for key, label_key, box_mask_key in self.key_iterator(d,self.label_keys,self.box_mask_keys):
            # self.push_transform(d, key)
            # first convert the mask back to the one with bg_label
            d[key], d[label_key] = self.converter(np.round(d[box_mask_key]) + self.bg_label)
            key_id += 1
        return d

# class BoxRandRotated(RandomizableTransform, MapTransform, InvertibleTransform):
#     """
#     Dictionary-based version :py:class:`monai.transforms.RandRotate`
#     Randomly rotates the input arrays.
#     Args:
#         keys: Keys to pick data for transformation.
#         range_x: Range of rotation angle in radians in the plane defined by the first and second axes.
#             If single number, angle is uniformly sampled from (-range_x, range_x).
#         range_y: Range of rotation angle in radians in the plane defined by the first and third axes.
#             If single number, angle is uniformly sampled from (-range_y, range_y).
#         range_z: Range of rotation angle in radians in the plane defined by the second and third axes.
#             If single number, angle is uniformly sampled from (-range_z, range_z).
#         prob: Probability of rotation.
#         keep_size: If it is False, the output shape is adapted so that the
#             input array is contained completely in the output.
#             If it is True, the output shape is the same as the input. Default is True.
#         mode: {``"bilinear"``, ``"nearest"``}
#             Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
#             See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
#             It also can be a sequence of string, each element corresponds to a key in ``keys``.
#         padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
#             Padding mode for outside grid values. Defaults to ``"border"``.
#             See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
#             It also can be a sequence of string, each element corresponds to a key in ``keys``.
#         align_corners: Defaults to False.
#             See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
#             It also can be a sequence of bool, each element corresponds to a key in ``keys``.
#         dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
#             If None, use the data type of input data. To be compatible with other modules,
#             the output data type is always ``np.float32``.
#             It also can be a sequence of dtype or None, each element corresponds to a key in ``keys``.
#         allow_missing_keys: don't raise exception if key is missing.
#     """

#     backend = RandRotate.backend

#     def __init__(
#         self,
#         box_keys: KeysCollection,
#         label_keys: KeysCollection,
#         box_mask_keys: KeysCollection,
#         image_key: KeysCollection,
#         ref_image_key: str,
#         min_fg_label: int = 0,
#         ellipse_mask: bool = False,
#         range_x: Union[Tuple[float, float], float] = 0.0,
#         range_y: Union[Tuple[float, float], float] = 0.0,
#         range_z: Union[Tuple[float, float], float] = 0.0,
#         prob: float = 0.1,
#         keep_size: bool = True,
#         mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
#         padding_mode: GridSamplePadModeSequence = GridSamplePadMode.BORDER,
#         align_corners: Union[Sequence[bool], bool] = False,
#         dtype: Union[Sequence[Union[DtypeLike, torch.dtype]], DtypeLike, torch.dtype] = np.float32,
#         allow_missing_keys: bool = False,
#     ) -> None:
#         MapTransform.__init__(self, keys, allow_missing_keys)
#         RandomizableTransform.__init__(self, prob)
#         self.box_to_mask = BoxToBoxMaskd(
#             box_keys=box_keys,
#             label_keys=label_keys,
#             box_mask_keys=box_mask_keys,
#             image_key=ref_image_keys,
#             min_fg_label=min_fg_label,
#             ellipse_mask=ellipse_mask)
#         self.mask_to_box = BoxToBoxMaskd(
#             box_keys=box_keys,
#             label_keys=label_keys,
#             box_mask_keys=box_mask_keys,
#             image_key=ref_image_keys,
#             min_fg_label=min_fg_label)
#         self.rotater = RandRotated(keys=image_key+box_mask_keys,
#             prob=1.,
#             range_x=range_x,range_y=range_y,range_z=range_z,
#             keep_size=keep_size,
#             mode=mode,padding_mode=padding_mode,align_corners=align_corners,dtype=dtype)

#     def set_random_state(
#         self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
#     ) -> "RandRotated":
#         super().set_random_state(seed, state)
#         self.rand_rotate.set_random_state(seed, state)
#         return self

#     def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
#         d = dict(data)
#         self.randomize(None)

#         # all the keys share the same random rotate angle
#         self.rand_rotate.randomize()

#         if self._do_transform:
#             d = self.box_to_mask(d)
#             d = self.rotater(d)
#             d = self.mask_to_box(d)
#         return d

class BoxRandCropForegroundd(Randomizable, MapTransform, InvertibleTransform):
    """
    Crop random fixed sized regions that contains foreground box.
    Suppose all the expected fields specified by `keys` have same shape,
    and add `patch_index` to the corresponding meta data.
    And will return a list of dictionaries for all the cropped images.
    If a dimension of the expected spatial size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected size,
    and the cropped results of several images may not have exactly the same shape.
    This function is not recommanded when the foreground objects are tiny, 
    as this function can not provide enough randomness in this case.
    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `data[label_key]` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        num_samples: number of samples (crop regions) to take in each list.
        fg_ratio: default 1. It will (crop fg_ratio*num_samples) patches with boxes in them for sure, 
            then crop (num_samples-fg_ratio*num_samples) patches randomly
        meta_keys: explicitly indicate the key of the corresponding meta data dictionary.
            used to add `patch_index` to the meta dict.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the meta data is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `key_{postfix}` to fetch the meta data according
            to the key data, default is `meta_dict`, the meta data is a dictionary object.
            used to add `patch_index` to the meta dict.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
        whole_box: Bool, default True, whether we prefer to contain the whole box in the cropped patch. 
            Even if True, it is still possible to get partial box if there are multiple boxes in the image
        mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                it also can be a sequence of string, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.
    """

    backend = RandCropByPosNegLabel.backend

    def __init__(
        self,
        keys: KeysCollection,
        box_key: str,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        num_samples: int = 1,
        fg_ratio: float = 1.0,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        allow_smaller: bool = False,
        whole_box: bool = True,
        mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = NumpyPadMode.CONSTANT,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.box_key = box_key
        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        self.num_samples = num_samples
        self.fg_num_samples = int(round(fg_ratio*num_samples))
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.centers: Optional[List[List[int]]] = None
        self.allow_smaller = allow_smaller
        self.whole_box = whole_box
        self.mode = ensure_tuple_rep(mode, len(self.keys))

    def randomize_fg(
        self,
        bbox: NdarrayOrTensor,
        image_size: NdarrayOrTensor,
    ) -> None:
        
        if self.fg_num_samples<=0:
            self.fg_centers = []
            return
            
        spatial_dims = len(image_size)
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().detach().numpy()

        fg_indices_ = []
        # if self.num_samples less than number of boxes, we sample box first to save time and memory
        if self.fg_num_samples>=bbox.shape[0]:
            sampled_box = list(range(bbox.shape[0]))
        else:
            sampled_box = np.random.permutation(bbox.shape[0])[:self.fg_num_samples]
        for bb in sampled_box:
            box = bbox[bb,:]
            fg_range_list = []                
            for axis in range(spatial_dims):                
                fg_range_min_axis = int(np.ceil(box[axis])) # box_start
                fg_range_max_axis = int(np.floor(box[axis+spatial_dims])) # box_stop
                if self.whole_box:
                    fg_range_min_axis = min(fg_range_min_axis, fg_range_max_axis-self.spatial_size[axis]//2-1)
                    fg_range_max_axis = max(fg_range_max_axis-1, fg_range_min_axis+self.spatial_size[axis]//2+1)
                else:
                    fg_range_min_axis = fg_range_min_axis-self.spatial_size[axis]//2+1
                    fg_range_max_axis = fg_range_max_axis+self.spatial_size[axis]//2-1
                fg_range_min_axis = max(fg_range_min_axis,0)
                fg_range_max_axis = min(fg_range_max_axis, image_size[axis]-1)
                fg_range_list.append( [*range(fg_range_min_axis, fg_range_max_axis) ] )
            if spatial_dims ==2:
                fg_unravel_index = np.meshgrid(fg_range_list[0], fg_range_list[1], indexing='ij')
            if spatial_dims ==3:
                fg_unravel_index = np.meshgrid(fg_range_list[0], fg_range_list[1], fg_range_list[2], indexing='ij')
            fg_unravel_index = [np.ravel(i) for i in fg_unravel_index]
            fg_indices_.append( np.ravel_multi_index(fg_unravel_index,image_size) )
            
        if len(fg_indices_)>0:
            fg_indices_ = np.concatenate(fg_indices_,axis=0)
            bg_indices_ = np.ones(1)*(-1)
            self.fg_centers = generate_pos_neg_label_crop_centers(
                self.spatial_size,
                self.fg_num_samples,
                1.,
                image_size,
                fg_indices_,
                bg_indices_,
                self.R,
                self.allow_smaller,
            )
        else:
            self.fg_centers = []
        

    def randomize(
        self,
        image_size: NdarrayOrTensor,
        num_rand_sample: int
    ) -> None:
        
        if num_rand_sample<=0:
            self.rand_centers = []
            return

        spatial_dims = len(image_size)

        fg_range_list = []               
        for axis in range(spatial_dims):                
            fg_range_min_axis = self.spatial_size[axis]//2
            fg_range_max_axis = max(self.spatial_size[axis]//2, image_size[axis]-self.spatial_size[axis]//2)
            fg_range_list.append( [fg_range_min_axis, fg_range_max_axis] )
        
        self.rand_centers = []
        for i in range(num_rand_sample):
            center = []
            for axis in range(spatial_dims):
                center.append(random.randint(fg_range_list[axis][0], fg_range_list[axis][1]+1))
            self.rand_centers.append(center)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        d = dict(data)
        bbox = d[self.box_key]
        label = d[self.label_key]
        spatial_dims = len(d[self.keys[0]].shape)-1
        self.spatial_size = ensure_tuple_rep(self.spatial_size, spatial_dims)
        # if image size smaller than crop size, pad image in that dimension
        for i in range(len(self.keys)):
            image_size = d[self.keys[i]].shape[1:]
            pad_size = [max(s1,s2)-s1 for s1,s2 in zip(image_size,self.spatial_size)]
            if max(pad_size)==0:
                continue
            pad_size = [[0,0]]+[[0,p] for p in pad_size] # pad in the bottom right so boxes will not be affected
            padder = Pad(pad_size,mode=self.mode[i])
            d[self.keys[i]] = padder(d[self.keys[i]])
        image_size = d[self.keys[0]].shape[1:]

        self.randomize_fg(bbox, image_size)
        self.randomize(image_size,num_rand_sample=self.num_samples-len(self.fg_centers))
        self.centers = self.rand_centers + self.fg_centers
        if not isinstance(self.spatial_size, tuple):
            raise ValueError("spatial_size must be a valid tuple.")
        if self.centers is None:
            raise ValueError("no available ROI centers to crop.")

        # initialize returned list with shallow copy to preserve key ordering
        results: List[Dict[Hashable, NdarrayOrTensor]] = [dict(d) for _ in range(self.num_samples)]

        for i, center in enumerate(self.centers):
            # fill in the extra keys with unmodified data
            cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)
            box_start = [max(s.start,0) for s in cropper.slices]
            box_end = [min(s.stop,image_size_a) for s,image_size_a in zip(cropper.slices,image_size)]
            box_slices = [slice(int(s), int(e)) for s,e in zip(box_start, box_end)]
            cropper = SpatialCrop(roi_slices=box_slices)
            for key in set(d.keys()).difference(set(self.keys)):
                results[i][key] = deepcopy(d[key])
            for key in self.key_iterator(d):
                img = d[key]
                orig_size = img.shape[1:]
                results[i][key] = cropper(img)
                self.push_transform(results[i], key, extra_info={"center": center}, orig_size=orig_size)
            boxcropper = BoxSpatialCropPad(roi_slices=box_slices)
            results[i][self.box_key], results[i][self.label_key] = boxcropper(bbox,label)
            self.push_transform(results[i], self.box_key, extra_info={"roi_slices": box_slices}, orig_size=orig_size)
            self.push_transform(results[i], self.label_key, extra_info={"roi_slices": box_slices}, orig_size=orig_size)
            # add `patch_index` to the meta data
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore

        return results

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        # TO DO: not done
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[TraceKeys.ORIG_SIZE])
            current_size = np.asarray(d[key].shape[1:])
            center = transform[TraceKeys.EXTRA_INFO]["center"]
            cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)  # type: ignore
            # get required pad to start and end
            pad_to_start = np.array([s.indices(o)[0] for s, o in zip(cropper.slices, orig_size)])
            pad_to_end = orig_size - current_size - pad_to_start
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            inverse_transform = BorderPad(pad)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d

class BoxCropForegroundd(MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.CropForeground`.
    Crop only the foreground object of the expected images.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    The valid part can be determined by any field in the data with `source_key`, for example:
    - Select values > 0 in image field as the foreground and crop on all fields specified by `keys`.
    - Select label = 3 in label field as the foreground to crop on all fields specified by `keys`.
    - Select label > 0 in the third channel of a One-Hot label field as the foreground to crop all `keys` fields.
    Users can define arbitrary function to select expected foreground from the whole source image or specified
    channels. And it can also add margin to every dim of the bounding box of foreground object.
    """

    # backend = CropForeground.backend

    def __init__(
        self,
        keys: KeysCollection,
        box_key: str,
        label_key: str,
        margin: Union[Sequence[int], int] = 0,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = NumpyPadMode.CONSTANT,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        allow_missing_keys: bool = False,
        **np_kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            box_key: data source to generate the bounding box of foreground.
            label_key: label for the boxes
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            allow_smaller: now always True, when computing box size with `margin`, whether allow the image size to be smaller
                than box size, default to `True`. if the margined size is bigger than image size, will pad with
                specified `mode`.
            k_divisible: make each spatial dimension to be divisible by k, default to 1.
                if `k_divisible` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                it also can be a sequence of string, each element corresponds to a key in ``keys``.
            start_coord_key: key to record the start coordinate of spatial bounding box for foreground.
            end_coord_key: key to record the end coordinate of spatial bounding box for foreground.
            allow_missing_keys: don't raise exception if key is missing.
            np_kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
                more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        """
        super().__init__(keys, allow_missing_keys)
        allow_smaller = True
        self.box_key = box_key
        self.label_key = label_key
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.cropper = BoxCropForeground(
            margin=margin,
            allow_smaller=allow_smaller,
            k_divisible=k_divisible,
            **np_kwargs,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        image_size = d[self.keys[0]].shape[1:]
        # allows ROI bigger than image, i.e., box_start might be negative
        box_start, box_end = self.cropper.compute_bounding_box(bbox=d[self.box_key],image_size=image_size) 
        box_slices = [slice(int(s), int(e)) for s,e in zip(box_start, box_end)]
        d[self.start_coord_key] = box_start
        d[self.end_coord_key] = box_end
        for key, m in self.key_iterator(d, self.mode):
            self.push_transform(d, key, extra_info={"box_start": box_start, "box_end": box_end})
            d[key] = self.cropper.crop_pad(img=d[key],box_start=box_start, box_end=box_end,mode=m)
        boxcropper = BoxSpatialCropPad(roi_slices=box_slices)
        d[self.box_key], d[self.label_key] = boxcropper(d[self.box_key],d[self.label_key])
        self.push_transform(d, self.box_key, extra_info={"roi_slices": box_slices})
        self.push_transform(d, self.label_key, extra_info={"roi_slices": box_slices})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        # TO DO: not done
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[TraceKeys.ORIG_SIZE])
            cur_size = np.asarray(d[key].shape[1:])
            extra_info = transform[TraceKeys.EXTRA_INFO]
            box_start = np.asarray(extra_info["box_start"])
            box_end = np.asarray(extra_info["box_end"])
            # first crop the padding part
            roi_start = np.maximum(-box_start, 0)
            roi_end = cur_size - np.maximum(box_end - orig_size, 0)

            d[key] = SpatialCrop(roi_start=roi_start, roi_end=roi_end)(d[key])

            # update bounding box to pad
            pad_to_start = np.maximum(box_start, 0)
            pad_to_end = orig_size - np.minimum(box_end, orig_size)
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            # second pad back the original size
            d[key] = BorderPad(pad)(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d

class BoxRandZoomd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dict-based version :py:class:`monai.transforms.RandZoom`.
    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of zooming.
        min_zoom: Min zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, min_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        max_zoom: Max zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, max_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        keep_size: Should keep original size (pad if needed), default is True.
        allow_missing_keys: don't raise exception if key is missing.
        kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
            more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    backend = RandZoom.backend

    def __init__(
        self,
        box_keys: KeysCollection,
        image_keys: KeysCollection, 
        prob: float = 0.1,
        min_zoom: Union[Sequence[float], float] = 0.9,
        max_zoom: Union[Sequence[float], float] = 1.1,
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        padding_mode: PadModeSequence = NumpyPadMode.EDGE,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        keep_size: bool = True,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        MapTransform.__init__(self, image_keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_zoom = RandZoom(prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, keep_size=keep_size, **kwargs)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.box_keys = box_keys
        self.keep_size = keep_size

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "BoxRandZoomd":
        super().set_random_state(seed, state)
        self.rand_zoom.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        first_key: Union[Hashable, List] = self.first_key(d)
        if first_key == []:
            return d

        self.randomize(None)

        # all the keys share the same random zoom factor
        self.rand_zoom.randomize(d[first_key])  # type: ignore
        spatial_dims = len(d[first_key].shape)-1
        for key, mode, padding_mode, align_corners in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners
        ):            
            if self._do_transform:
                orig_image_size = d[key].shape[1:]
                d[key] = self.rand_zoom(
                    d[key], mode=mode, padding_mode=padding_mode, align_corners=align_corners, randomize=False
                )
                
            self.push_transform(
                d,
                key,
                extra_info={
                    "zoom": self.rand_zoom._zoom,
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
                },
            )
                    
        for box_key in self.box_keys:
            if self._do_transform:
                d[box_key] = BoxZoom(zoom=self.rand_zoom._zoom,keep_size=self.keep_size)(d[box_key],orig_image_size=orig_image_size)
                self.push_transform(
                    d,
                    box_key,
                    extra_info={
                        "zoom": self.rand_zoom._zoom,
                    },
                )
        
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # Create inverse transform
                zoom = np.array(transform[TraceKeys.EXTRA_INFO]["zoom"])
                mode = transform[TraceKeys.EXTRA_INFO]["mode"]
                padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
                align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
                inverse_transform = Zoom(zoom=(1 / zoom).tolist(), keep_size=self.rand_zoom.keep_size)
                # Apply inverse
                orig_image_size = d[key].shape[1:]
                d[key] = inverse_transform(
                    d[key],
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=None if align_corners == TraceKeys.NONE else align_corners,
                )
                # Size might be out by 1 voxel so pad
                d[key] = SpatialPad(transform[TraceKeys.ORIG_SIZE], mode="edge")(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        for box_key in self.box_keys:
            transform = self.get_most_recent_transform(d, box_key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # Create inverse transform
                zoom = np.array(transform[TraceKeys.EXTRA_INFO]["zoom"])
                box_inverse_transform = BoxZoom(zoom=(1 / zoom).tolist(), keep_size=self.rand_zoom.keep_size)
                d[box_key] = box_inverse_transform(
                    d[box_key], orig_image_size=orig_image_size
                )
            # Remove the applied transform
            self.pop_transform(d, box_key)

        return 

BoxConvertToStandardD = BoxConvertToStandardDict = BoxConvertToStandardd
BoxConvertModeD = BoxConvertModeDict = BoxConvertModed
BoxClipToImageD = BoxClipToImageDict = BoxClipToImaged
BoxToImageCoordinateD = BoxToImageCoordinateDict= BoxToImageCoordinated
BoxMaskToBoxD = BoxMaskToBoxDict = BoxMaskToBoxd
BoxToBoxMaskD = BoxToBoxMaskDict = BoxToBoxMaskd
BoxCropForegroundD = BoxCropForegroundDict = BoxCropForegroundd
BoxRandCropForegroundD = BoxRandCropForegroundDict = BoxRandCropForegroundd
BoxRandZoomD = BoxRandZoomDict = BoxRandZoomd
BoxRandFlipD = BoxRandFlipDict = BoxRandFlipd
BoxRandSwapAxesD = BoxRandSwapAxesDict = BoxRandSwapAxesd

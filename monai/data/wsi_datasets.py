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

import inspect
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from monai.data import Dataset
from monai.data.utils import iter_patch_position
from monai.data.wsi_reader import BaseWSIReader, WSIReader
from monai.transforms import ForegroundMask, Randomizable, apply_transform
from monai.utils import CommonKeys, ProbMapKeys, ensure_tuple_rep
from monai.utils.enums import WSIPatchKeys

__all__ = ["PatchWSIDataset", "SlidingPatchWSIDataset", "MaskedPatchWSIDataset"]


class PatchWSIDataset(Dataset):
    """
    This dataset extracts patches from whole slide images (without loading the whole image)
    It also reads labels for each patch and provides each patch with its associated class labels.

    Args:
        data: the list of input samples including image, location, and label (see the note below for more details).
        size: the size of patch to be extracted from the whole slide image.
        level: the level at which the patches to be extracted (default to 0).
        transform: transforms to be executed on input data.
        include_label: whether to load and include labels in the output
        center_location: whether the input location information is the position of the center of the patch
        additional_meta_keys: the list of keys for items to be copied to the output metadata from the input data
        reader: the module to be used for loading whole slide imaging. If `reader` is

            - a string, it defines the backend of `monai.data.WSIReader`. Defaults to cuCIM.
            - a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader.
            - an instance of a a class inherited from `BaseWSIReader`, it is set as the wsi_reader.

        kwargs: additional arguments to pass to `WSIReader` or provided whole slide reader class

    Note:
        The input data has the following form as an example:

        .. code-block:: python

            [
                {"image": "path/to/image1.tiff", "patch_location": [200, 500], "label": 0},
                {"image": "path/to/image2.tiff", "patch_location": [100, 700], "patch_size": [20, 20], "level_size": 2, "label": 1}
            ]

    """

    def __init__(
        self,
        data: Sequence,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_level: Optional[int] = None,
        transform: Optional[Callable] = None,
        include_label: bool = True,
        center_location: bool = True,
        additional_meta_keys: Optional[Sequence[str]] = None,
        reader="cuCIM",
        **kwargs,
    ):
        super().__init__(data, transform)

        # Ensure patch size is a two dimensional tuple
        if patch_size is None:
            self.patch_size = None
        else:
            self.patch_size = ensure_tuple_rep(patch_size, 2)

        # Create a default level that override all levels if it is not None
        self.patch_level = patch_level
        # Set the default WSIReader's level to 0 if level is not provided
        if patch_level is None:
            patch_level = 0

        # Setup the WSI reader
        self.wsi_reader: Union[WSIReader, BaseWSIReader]
        if isinstance(reader, str):
            self.wsi_reader = WSIReader(backend=reader, level=patch_level, **kwargs)
        elif inspect.isclass(reader) and issubclass(reader, BaseWSIReader):
            self.wsi_reader = reader(level=patch_level, **kwargs)
        elif isinstance(reader, BaseWSIReader):
            self.wsi_reader = reader
        else:
            raise ValueError(f"Unsupported reader type: {reader}.")
        self.backend = self.wsi_reader.backend

        self.include_label = include_label
        self.center_location = center_location
        self.additional_meta_keys = additional_meta_keys or []

        # Initialized an empty whole slide image object dict
        self.wsi_object_dict: Dict = {}

    def _get_wsi_object(self, sample: Dict):
        image_path = sample[CommonKeys.IMAGE]
        if image_path not in self.wsi_object_dict:
            self.wsi_object_dict[image_path] = self.wsi_reader.read(image_path)
        return self.wsi_object_dict[image_path]

    def _get_label(self, sample: Dict):
        return np.array(sample[CommonKeys.LABEL], dtype=np.float32)

    def _get_location(self, sample: Dict):
        if self.center_location:
            size = self._get_size(sample)
            return [sample[WSIPatchKeys.LOCATION][i] - size[i] // 2 for i in range(len(size))]
        else:
            return sample[WSIPatchKeys.LOCATION]

    def _get_level(self, sample: Dict):
        if self.patch_level is None:
            return sample.get(WSIPatchKeys.LEVEL, 0)
        return self.patch_level

    def _get_size(self, sample: Dict):
        if self.patch_size is None:
            return ensure_tuple_rep(sample.get(WSIPatchKeys.SIZE), 2)
        return self.patch_size

    def _get_data(self, sample: Dict):
        # Don't store OpenSlide objects to avoid issues with OpenSlide internal cache
        if self.backend == "openslide":
            self.wsi_object_dict = {}
        wsi_obj = self._get_wsi_object(sample)
        location = self._get_location(sample)
        level = self._get_level(sample)
        size = self._get_size(sample)
        return self.wsi_reader.get_data(wsi=wsi_obj, location=location, size=size, level=level)

    def _transform(self, index: int):
        # Get a single entry of data
        sample: Dict = self.data[index]

        # Extract patch image and associated metadata
        image, metadata = self._get_data(sample)
        output = {CommonKeys.IMAGE: image, CommonKeys.METADATA: metadata}

        # Include label in the output
        if self.include_label:
            output[CommonKeys.LABEL] = self._get_label(sample)

        for key in self.additional_meta_keys:
            metadata[key] = sample[key]

        # Apply transforms and return it
        return apply_transform(self.transform, output) if self.transform else output


class SlidingPatchWSIDataset(Randomizable, PatchWSIDataset):
    """
    This dataset extracts patches from whole slide images (without loading the whole image)
    It also reads labels for each patch and provides each patch with its associated class labels.

    Args:
        data: the list of input samples including image, location, and label (see the note below for more details).
        size: the size of patch to be extracted from the whole slide image.
        level: the level at which the patches to be extracted (default to 0).
        offset: the offset of image to extract patches (the starting position of the upper left patch).
        offset_limits: if offset is set to "random", a tuple of integers defining the lower and upper limit of the
            random offset for all dimensions, or a tuple of tuples that defines the limits for each dimension.
        overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
            If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
        transform: transforms to be executed on input data.
        reader: the module to be used for loading whole slide imaging. Defaults to cuCIM. If `reader` is

            - a string, it defines the backend of `monai.data.WSIReader`.
            - a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader,
            - an instance of a a class inherited from `BaseWSIReader`, it is set as the wsi_reader.

        seed: random seed to randomly generate offsets. Defaults to 0.
        kwargs: additional arguments to pass to `WSIReader` or provided whole slide reader class

    Note:
        The input data has the following form as an example:

        .. code-block:: python

            [
                {"image": "path/to/image1.tiff"},
                {"image": "path/to/image2.tiff", "patch_size": [20, 20], "patch_level": 2}
            ]

    """

    def __init__(
        self,
        data: Sequence,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_level: Optional[int] = None,
        mask_level: int = 0,
        overlap: Union[Tuple[float, float], float] = 0.0,
        offset: Union[Tuple[int, int], int, str] = (0, 0),
        offset_limits: Optional[Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, int]]] = None,
        transform: Optional[Callable] = None,
        include_label: bool = False,
        center_location: bool = False,
        additional_meta_keys: Sequence[str] = (
            ProbMapKeys.LOCATION,
            ProbMapKeys.SIZE,
            ProbMapKeys.COUNT,
        ),
        reader="cuCIM",
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            data=data,
            patch_size=patch_size,
            patch_level=patch_level,
            transform=transform,
            include_label=include_label,
            center_location=center_location,
            additional_meta_keys=additional_meta_keys,
            reader=reader,
            **kwargs,
        )
        self.mask_level = mask_level
        self.overlap = overlap
        self.set_random_state(seed)
        # Set the offset config
        self.random_offset = False
        if isinstance(offset, str):
            if offset == "random":
                self.random_offset = True
                self.offset_limits: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
                if offset_limits is None:
                    self.offset_limits = None
                elif isinstance(offset_limits, tuple):
                    if isinstance(offset_limits[0], int):
                        self.offset_limits = (offset_limits, offset_limits)
                    elif isinstance(offset_limits[0], tuple):
                        self.offset_limits = offset_limits
                    else:
                        raise ValueError(
                            "The offset limits should be either a tuple of integers or tuple of tuple of integers."
                        )
                else:
                    raise ValueError("The offset limits should be a tuple.")
            else:
                raise ValueError(
                    f'Invalid string for offset "{offset}". It should be either "random" as a string,'
                    "an integer, or a tuple of integers defining the offset."
                )
        else:
            self.offset = ensure_tuple_rep(offset, 2)

        # Create single sample for each patch (in a sliding window manner)
        self.data = []
        self.image_data = data
        for sample in self.image_data:
            patch_samples = self._evaluate_patch_locations(sample)
            self.data.extend(patch_samples)

    def _get_offset(self, sample):
        if self.random_offset:
            if self.offset_limits is None:
                offset_limits = tuple((-s, s) for s in self._get_size(sample))
            else:
                offset_limits = self.offset_limits
            return tuple(self.R.randint(low, high) for low, high in offset_limits)
        return self.offset

    def _evaluate_patch_locations(self, sample):
        """Calculate the location for each patch in a sliding-window manner"""
        patch_size = self._get_size(sample)
        patch_level = self._get_level(sample)
        wsi_obj = self._get_wsi_object(sample)

        # calculate the locations
        wsi_size = self.wsi_reader.get_size(wsi_obj, 0)
        mask_ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, self.mask_level)
        patch_ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, patch_level)
        patch_size_0 = np.array([p * patch_ratio for p in patch_size])  # patch size at level 0
        offset = self._get_offset(sample)
        patch_locations = np.array(
            list(
                iter_patch_position(
                    image_size=wsi_size, patch_size=patch_size_0, start_pos=offset, overlap=self.overlap, padded=False
                )
            )
        )
        # convert locations to mask_location
        mask_locations = np.round((patch_locations + patch_size_0 // 2) / float(mask_ratio))

        # fill out samples with location and metadata
        sample[WSIPatchKeys.SIZE] = patch_size
        sample[WSIPatchKeys.LEVEL] = patch_level
        sample[ProbMapKeys.COUNT] = len(patch_locations)
        sample[ProbMapKeys.SIZE] = np.array(self.wsi_reader.get_size(wsi_obj, self.mask_level))
        return [
            {**sample, WSIPatchKeys.LOCATION: np.array(loc), ProbMapKeys.LOCATION: mask_loc}
            for loc, mask_loc in zip(patch_locations, mask_locations)
        ]


class MaskedPatchWSIDataset(PatchWSIDataset):
    """
    This dataset extracts patches from whole slide images at the locations where foreground mask
    at a given level is non-zero.

    Args:
        data: the list of input samples including image, location, and label (see the note below for more details).
        size: the size of patch to be extracted from the whole slide image.
        level: the level at which the patches to be extracted (default to 0).
        mask_level: the resolution level at which the mask is created.
        transform: transforms to be executed on input data.
        include_label: whether to load and include labels in the output
        center_location: whether the input location information is the position of the center of the patch
        additional_meta_keys: the list of keys for items to be copied to the output matadata from the input data
        reader: the module to be used for loading whole slide imaging. Defaults to cuCIM. If `reader` is

            - a string, it defines the backend of `monai.data.WSIReader`.
            - a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader,
            - an instance of a a class inherited from `BaseWSIReader`, it is set as the wsi_reader.

        kwargs: additional arguments to pass to `WSIReader` or provided whole slide reader class

    Note:
        The input data has the following form as an example:

        .. code-block:: python

            [
                {"image": "path/to/image1.tiff"},
                {"image": "path/to/image2.tiff", "patch_size": [20, 20], "patch_level": 2}
            ]

    """

    def __init__(
        self,
        data: Sequence,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_level: Optional[int] = None,
        mask_level: int = 7,
        transform: Optional[Callable] = None,
        include_label: bool = False,
        center_location: bool = False,
        additional_meta_keys: Sequence[str] = (
            ProbMapKeys.LOCATION,
            ProbMapKeys.SIZE,
            ProbMapKeys.COUNT,
        ),
        reader="cuCIM",
        **kwargs,
    ):
        super().__init__(
            data=data,
            patch_size=patch_size,
            patch_level=patch_level,
            transform=transform,
            include_label=include_label,
            center_location=center_location,
            additional_meta_keys=additional_meta_keys,
            reader=reader,
            **kwargs,
        )
        self.mask_level = mask_level
        # Create single sample for each patch (in a sliding window manner)
        self.data = []
        for sample in data:
            patch_samples = self._evaluate_patch_coordinates(sample)
            self.data.extend(patch_samples)

    def _evaluate_patch_coordinates(self, sample):
        """Calculate the location for each patch based on the mask at different resolution level"""
        patch_size = self._get_size(sample)
        patch_level = self._get_level(sample)
        wsi_obj = self._get_wsi_object(sample)

        # load the entire image at level=mask_level
        wsi, _ = self.wsi_reader.get_data(wsi_obj, level=self.mask_level)

        # create the foreground tissue mask and get all indices for non-zero pixels
        mask = np.squeeze(ForegroundMask(hsv_threshold={"S": "otsu"})(wsi))
        mask_locations = np.vstack(mask.nonzero()).T

        # convert mask locations to image locations at level=0
        mask_ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, self.mask_level)
        patch_ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, patch_level)
        patch_size_0 = np.array([p * patch_ratio for p in patch_size])  # patch size at level 0
        patch_locations = np.round((mask_locations + 0.5) * float(mask_ratio) - patch_size_0 // 2).astype(int)

        # fill out samples with location and metadata
        sample[WSIPatchKeys.SIZE] = patch_size
        sample[WSIPatchKeys.LEVEL] = patch_level
        sample[ProbMapKeys.COUNT] = len(patch_locations)
        sample[ProbMapKeys.SIZE] = np.array(self.wsi_reader.get_size(wsi_obj, self.mask_level))
        return [
            {**sample, WSIPatchKeys.LOCATION: np.array(loc), ProbMapKeys.LOCATION: mask_loc}
            for loc, mask_loc in zip(patch_locations, mask_locations)
        ]

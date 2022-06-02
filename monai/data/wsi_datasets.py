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
from monai.transforms import Randomizable, apply_transform
from monai.utils import ensure_tuple_rep

__all__ = ["PatchWSIDataset", "SlidingPatchWSIDataset"]


class PatchWSIDataset(Dataset):
    """
    This dataset extracts patches from whole slide images (without loading the whole image)
    It also reads labels for each patch and provides each patch with its associated class labels.

    Args:
        data: the list of input samples including image, location, and label (see the note below for more details).
        size: the size of patch to be extracted from the whole slide image.
        level: the level at which the patches to be extracted (default to 0).
        transform: transforms to be executed on input data.
        reader: the module to be used for loading whole slide imaging. If `reader` is

            - a string, it defines the backend of `monai.data.WSIReader`. Defaults to cuCIM.
            - a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader.
            - an instance of a a class inherited from `BaseWSIReader`, it is set as the wsi_reader.

        kwargs: additional arguments to pass to `WSIReader` or provided whole slide reader class

    Note:
        The input data has the following form as an example:

        .. code-block:: python

            [
                {"image": "path/to/image1.tiff", "location": [200, 500], "label": 0},
                {"image": "path/to/image2.tiff", "location": [100, 700], "size": [20, 20], "level": 2, "label": 1}
            ]

    """

    def __init__(
        self,
        data: Sequence,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        level: Optional[int] = None,
        transform: Optional[Callable] = None,
        reader="cuCIM",
        **kwargs,
    ):
        super().__init__(data, transform)

        # Ensure patch size is a two dimensional tuple
        if size is None:
            self.size = None
        else:
            self.size = ensure_tuple_rep(size, 2)

        # Create a default level that override all levels if it is not None
        self.level = level
        # Set the default WSIReader's level to 0 if level is not provided
        if level is None:
            level = 0

        # Setup the WSI reader
        self.wsi_reader: Union[WSIReader, BaseWSIReader]
        self.backend = ""
        if isinstance(reader, str):
            self.backend = reader.lower()
            self.wsi_reader = WSIReader(backend=self.backend, level=level, **kwargs)
        elif inspect.isclass(reader) and issubclass(reader, BaseWSIReader):
            self.wsi_reader = reader(level=level, **kwargs)
        elif isinstance(reader, BaseWSIReader):
            self.wsi_reader = reader
        else:
            raise ValueError(f"Unsupported reader type: {reader}.")

        # Initialized an empty whole slide image object dict
        self.wsi_object_dict: Dict = {}

    def _get_wsi_object(self, sample: Dict):
        image_path = sample["image"]
        if image_path not in self.wsi_object_dict:
            self.wsi_object_dict[image_path] = self.wsi_reader.read(image_path)
        return self.wsi_object_dict[image_path]

    def _get_label(self, sample: Dict):
        return np.array(sample["label"], dtype=np.float32)

    def _get_location(self, sample: Dict):
        size = self._get_size(sample)
        return [sample["location"][i] - size[i] // 2 for i in range(len(size))]

    def _get_level(self, sample: Dict):
        if self.level is None:
            return sample.get("level", 0)
        return self.level

    def _get_size(self, sample: Dict):
        if self.size is None:
            return ensure_tuple_rep(sample.get("size"), 2)
        return self.size

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

        # Get the label
        label = self._get_label(sample)

        # Apply transforms and output
        output = {"image": image, "label": label, "metadata": metadata}
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

        map_level: the resolution level at which the output map is created.
        seed: random seed to randomly generate offsets. Defaults to 0.
        kwargs: additional arguments to pass to `WSIReader` or provided whole slide reader class

    Note:
        The input data has the following form as an example:

        .. code-block:: python

            [
                {"image": "path/to/image1.tiff"},
                {"image": "path/to/image2.tiff", "size": [20, 20], "level": 2}
            ]

    """

    def __init__(
        self,
        data: Sequence,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        level: Optional[int] = None,
        overlap: Union[Tuple[float, float], float] = 0.0,
        offset: Union[Tuple[int, int], int, str] = (0, 0),
        offset_limits: Optional[Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, int]]] = None,
        transform: Optional[Callable] = None,
        reader="cuCIM",
        map_level: int = 0,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(data=[], size=size, level=level, transform=transform, reader=reader, **kwargs)
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

        self.map_level = map_level
        # Create single sample for each patch (in a sliding window manner)
        self.data = []
        self.image_data = data
        for sample in self.image_data:
            patch_samples = self._evaluate_patch_coordinates(sample)
            self.data.extend(patch_samples)

    def _get_offset(self, sample):
        if self.random_offset:
            if self.offset_limits is None:
                offset_limits = tuple((-s, s) for s in self._get_size(sample))
            else:
                offset_limits = self.offset_limits
            return tuple(self.R.randint(low, high) for low, high in offset_limits)
        return self.offset

    def _evaluate_patch_coordinates(self, sample):
        """Define the location for each patch in a sliding-window manner"""
        patch_size = self._get_size(sample)
        level = self._get_level(sample)
        offset = self._get_offset(sample)

        wsi_obj = self._get_wsi_object(sample)
        wsi_size = self.wsi_reader.get_size(wsi_obj, 0)
        ratio = self.wsi_reader.get_downsample_ratio(wsi_obj, level)
        patch_size_ = tuple(p * ratio for p in patch_size)  # patch size at level 0
        locations = list(
            iter_patch_position(
                image_size=wsi_size, patch_size=patch_size_, start_pos=offset, overlap=self.overlap, padded=False
            )
        )
        n_patches = len(locations)
        sample["size"] = np.array(patch_size)
        sample["level"] = level
        sample["num_patches"] = n_patches
        sample["map_level"] = self.map_level
        sample["map_size"] = np.array(self.wsi_reader.get_size(wsi_obj, self.map_level))
        return [
            {**sample, "location": np.array(loc), "map_center": self.downsample_center(loc, patch_size, ratio)}
            for loc in locations
        ]

    def downsample_center(self, location: Tuple[int, int], patch_size: Tuple[int, int], ratio: float) -> np.ndarray:
        """
        For a given location at level=0, evaluate the corresponding center position of patch at level=`level`
        """
        center_location = [int((l + p // 2) / ratio) for l, p in zip(location, patch_size)]
        return np.array(center_location)

    def _get_location(self, sample: Dict):
        return sample["location"]

    def _transform(self, index: int):
        # Get a single entry of data
        sample: Dict = self.data[index]
        # Extract patch image and associated metadata
        image, metadata = self._get_data(sample)
        for k in ["num_patches", "map_level", "map_size", "map_center"]:
            metadata[k] = sample[k]
        # Create put all patch information together and apply transforms
        patch = {"image": image, "metadata": metadata}
        return apply_transform(self.transform, patch) if self.transform else patch

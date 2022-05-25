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
from monai.data.wsi_reader import BaseWSIReader, WSIReader
from monai.transforms import apply_transform
from monai.utils import ensure_tuple_rep

__all__ = ["PatchWSIDataset"]


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

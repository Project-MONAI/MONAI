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

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from monai.data import Dataset
from monai.data.wsi_reader import WSIReader
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
        reader_name: the name of library to be used for loading whole slide imaging, as the backend of `monai.data.WSIReader`
            Defaults to CuCIM.

    Note:
        The input data has the following form as an example:

        .. code-block:: python

            [
                {"image": "path/to/image1.tiff", "location": [200, 500], "label": 0},
                {"image": "path/to/image2.tiff", "location": [100, 700], "label": 1}
            ]

    """

    def __init__(
        self,
        data: List,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        level: Optional[int] = None,
        transform: Optional[Callable] = None,
        reader_name: str = "cuCIM",
    ):
        super().__init__(data, transform)

        # Ensure patch size is a two dimensional tuple
        if size is None:
            self.size = None
        else:
            self.size = ensure_tuple_rep(size, 2)

        # Create a default level
        self.level = level

        # Setup the WSI reader backend
        self.reader_name = reader_name.lower()
        self.image_reader = WSIReader(backend=self.reader_name)

        # Initialized an empty whole slide image object dict
        self.wsi_object_dict: Dict = {}

    def _get_wsi_object(self, sample: Dict):
        image_path = sample["image"]
        if image_path not in self.wsi_object_dict:
            self.wsi_object_dict[image_path] = self.image_reader.read(image_path)
        return self.wsi_object_dict[image_path]

    def _get_label(self, sample: Dict):
        return np.array(sample["label"], dtype=np.float32)

    def _get_location(self, sample: Dict):
        size = self._get_size(sample)
        return [sample["location"][i] - size[i] // 2 for i in range(len(size))]

    def _get_level(self, sample: Dict):
        if self.level is None:
            return sample.get("level")
        return self.level

    def _get_size(self, sample: Dict):
        if self.size is None:
            return ensure_tuple_rep(sample.get("size"), 2)
        return self.size

    def _get_data(self, sample: Dict):
        if self.reader_name == "openslide":
            self.wsi_object_dict = {}
        wsi_obj = self._get_wsi_object(sample)
        location = self._get_location(sample)
        level = self._get_level(sample)
        size = self._get_size(sample)
        return self.image_reader.get_data(wsi=wsi_obj, location=location, size=size, level=level)

    def _transform(self, index: int):
        # Get a single entry of data
        sample: Dict = self.data[index]
        # Extract patch image and associated metadata
        image, metadata = self._get_data(sample)
        # Get the label
        label = self._get_label(sample)

        # Create put all patch information together and apply transforms
        patch = {"image": image, "label": label, "metadata": metadata}
        return apply_transform(self.transform, patch) if self.transform else patch

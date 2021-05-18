# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from collections import OrderedDict
from typing import Dict, Optional, Union

import numpy as np
import torch

from monai.utils import ImageMetaKey as Key


class CSVSaver:
    """
    Save the data in a dictionary format cache, and write to a CSV file finally.
    Typically, the data can be classification predictions, call `save` for single data
    or call `save_batch` to save a batch of data together, and call `finalize` to write
    the cached data into CSV file. If no meta data provided, use index from 0 to save data.
    Note that this saver can't support multi-processing because it reads / writes single
    CSV file and can't guarantee the data order in multi-processing situation.

    """

    def __init__(
        self,
        output_dir: str = "./",
        filename: str = "predictions.csv",
        overwrite: bool = True,
        flush: bool = False,
    ) -> None:
        """
        Args:
            output_dir: output CSV file directory.
            filename: name of the saved CSV file name.
            overwrite: whether to overwriting existing CSV file content, if True, will clear the file before saving.
                otherwise, will apend new content to the CSV file.
            flush: whether to write the cache data to CSV file immediately when `save_batch` and clear the cache.
                default to False.

        """
        self.output_dir = output_dir
        self._cache_dict: OrderedDict = OrderedDict()
        if not (isinstance(filename, str) and filename[-4:] == ".csv"):
            warnings.warn("CSV filename is not a string ends with '.csv'.")
        self._filepath = os.path.join(output_dir, filename)
        if os.path.exists(self._filepath) and overwrite:
            os.remove(self._filepath)

        self.flush = flush
        self._data_index = 0

    def finalize(self) -> None:
        """
        Writes the cached dict to a csv

        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(self._filepath, "a") as f:
            for k, v in self._cache_dict.items():
                f.write(k)
                for result in v.flatten():
                    f.write("," + str(result))
                f.write("\n")
        # clear cache content after writing
        self.reset_cache()

    def save(self, data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
        """Save data into the cache dictionary. The metadata should have the following key:
            - ``'filename_or_obj'`` -- save the data corresponding to file name or object.
        If meta_data is None, use the default index from 0 to save data instead.

        Args:
            data: target data content that save into cache.
            meta_data: the meta data information corresponding to the data.

        """
        save_key = meta_data[Key.FILENAME_OR_OBJ] if meta_data else str(self._data_index)
        self._data_index += 1
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if not isinstance(data, np.ndarray):
            raise AssertionError
        self._cache_dict[save_key] = data.astype(np.float32)

    def save_batch(self, batch_data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
        """Save a batch of data into the cache dictionary.

        Args:
            batch_data: target batch data content that save into cache.
            meta_data: every key-value in the meta_data is corresponding to 1 batch of data.

        """
        for i, data in enumerate(batch_data):  # save a batch of files
            self.save(data, {k: meta_data[k][i] for k in meta_data} if meta_data else None)

        if self.flush:
            self.finalize()

    def get_cache(self) -> OrderedDict:
        """Get the cache dictionary, key is filename and value is the corresponding data"""

        return self._cache_dict

    def reset_cache(self) -> None:
        """Clear the cache dictionary content"""
        self._cache_dict.clear()

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

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import numpy as np
import torch

from monai.config import DtypeLike
from monai.data.utils import create_file_basename
from monai.utils import GridSampleMode, GridSamplePadMode, optional_import
from monai.utils import ImageMetaKey as Key

itk, _ = optional_import("itk", allow_namespace_pkg=True)


class ImageWriter(ABC):
    def __init__(
        self,
        output_dir: str = "./",
        output_postfix: str = "seg",
        output_ext: str = ".nii.gz",
        squeeze_end_dims: bool = True,
        data_root_dir: str = "",
        separate_folder: bool = True,
        print_log: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.output_ext = output_ext
        self._data_index = 0
        self.squeeze_end_dims = squeeze_end_dims
        self.data_root_dir = data_root_dir
        self.separate_folder = separate_folder
        self.print_log = print_log

    def save(self, data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
        filename = meta_data[Key.FILENAME_OR_OBJ] if meta_data else str(self._data_index)
        self._data_index += 1
        patch_index = meta_data.get(Key.PATCH_INDEX, None) if meta_data else None

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        path = create_file_basename(
            postfix=self.output_postfix,
            input_file_name=filename,
            folder_path=self.output_dir,
            data_root_dir=self.data_root_dir,
            separate_folder=self.separate_folder,
            patch_index=patch_index,
        )
        path = f"{path}{self.output_ext}"

        # change data to "channel last" format and write to file
        data = np.moveaxis(np.asarray(data), 0, -1)

        # if desired, remove trailing singleton dimensions
        if self.squeeze_end_dims:
            while data.shape[-1] == 1:
                data = np.squeeze(data, -1)

        self.write(data=data, meta_data=meta_data, filename=path)

        if self.print_log:
            print(f"file written: {path}.")

    def save_batch(self, batch_data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
        for i, data in enumerate(batch_data):  # save a batch of files
            self.save(data=data, meta_data={k: meta_data[k][i] for k in meta_data} if meta_data is not None else None)

    @abstractmethod
    def write(self, data, meta_data, filename):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class ITKWriter(ImageWriter):
    def __init__(
        self,
        resample: bool = True,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike = np.float32,
    ) -> None:
        self.resample = resample
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.align_corners = align_corners
        self.dtype = dtype
        self.output_dtype = output_dtype

    def write(self, data, meta_data, filename):
        pass

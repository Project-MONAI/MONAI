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
from monai.data.utils import (
    adjust_orientation_by_affine,
    adjust_spatial_shape_by_affine,
    create_file_basename,
    to_affine_nd,
)
from monai.utils import GridSampleMode, GridSamplePadMode
from monai.utils import ImageMetaKey as Key
from monai.utils import optional_import

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

    def write(self, data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
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

        self._write_file(data=data, filename=path, meta_data=meta_data)

        if self.print_log:
            print(f"file written: {path}.")

    def write_batch(self, batch_data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
        for i, data in enumerate(batch_data):  # write a batch of data to files
            self.write(data=data, meta_data={k: meta_data[k][i] for k in meta_data} if meta_data is not None else None)

    @abstractmethod
    def _write_file(self, data: np.ndarray, filename: str, meta_data: Optional[Dict] = None):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class ITKWriter(ImageWriter):
    def __init__(
        self,
        output_dir: str = "./",
        output_postfix: str = "seg",
        output_ext: str = ".dcm",
        squeeze_end_dims: bool = True,
        data_root_dir: str = "",
        separate_folder: bool = True,
        print_log: bool = True,
        resample: bool = True,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike = np.float32,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            output_postfix=output_postfix,
            output_ext=output_ext,
            squeeze_end_dims=squeeze_end_dims,
            data_root_dir=data_root_dir,
            separate_folder=separate_folder,
            print_log=print_log,
        )
        self.resample = resample
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.align_corners = align_corners
        self.dtype = dtype
        self.output_dtype = output_dtype

    def _write_file(self, data: np.ndarray, filename: str, meta_data: Optional[Dict] = None):
        target_affine = meta_data.get("original_affine", None) if meta_data else None
        affine = meta_data.get("affine", None) if meta_data else None
        spatial_shape = meta_data.get("spatial_shape", None) if meta_data else None

        if not isinstance(data, np.ndarray):
            raise AssertionError("input data must be numpy array.")
        dtype = self.dtype or data.dtype
        sr = min(data.ndim, 3)
        if affine is None:
            affine = np.eye(4, dtype=np.float64)
        affine = to_affine_nd(sr, affine)

        if target_affine is None:
            target_affine = affine
        target_affine = to_affine_nd(sr, target_affine)

        if not np.allclose(affine, target_affine, atol=1e-3):
            data, affine = adjust_orientation_by_affine(data=data, affine=affine, target_affine=target_affine)
            if self.resample:
                data, affine = adjust_spatial_shape_by_affine(
                    data=data,
                    affine=affine,
                    target_affine=target_affine,
                    output_spatial_shape=spatial_shape,
                    mode=self.mode,
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners,
                    dtype=dtype,
                )

        itk_np_view = itk.image_view_from_array(data.astype(self.output_dtype))
        # TODO: need to set affine matrix into file header
        # itk_np_view.SetMatrix(to_affine_nd(3, affine))
        itk.imwrite(itk_np_view, filename)

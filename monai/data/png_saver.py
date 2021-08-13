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

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from monai.data.png_writer import write_png
from monai.data.utils import create_file_basename
from monai.utils import ImageMetaKey as Key
from monai.utils import InterpolateMode, look_up_option


class PNGSaver:
    """
    Save the data as png file, it can support single data content or a batch of data.
    Typically, the data can be segmentation predictions, call `save` for single data
    or call `save_batch` to save a batch of data together.
    The name of saved file will be `{input_image_name}_{output_postfix}{output_ext}`,
    where the input image name is extracted from the provided meta data dictionary.
    If no meta data provided, use index from 0 as the filename prefix.

    """

    def __init__(
        self,
        output_dir: Union[Path, str] = "./",
        output_postfix: str = "seg",
        output_ext: str = ".png",
        resample: bool = True,
        mode: Union[InterpolateMode, str] = InterpolateMode.NEAREST,
        scale: Optional[int] = None,
        data_root_dir: str = "",
        separate_folder: bool = True,
        print_log: bool = True,
    ) -> None:
        """
        Args:
            output_dir: output image directory.
            output_postfix: a string appended to all output file names.
            output_ext: output file extension name.
            resample: whether to resample and resize if providing spatial_shape in the metadata.
            mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``"nearest"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
            scale: {``255``, ``65535``} postprocess data by clipping to [0, 1] and scaling
                [0, 255] (uint8) or [0, 65535] (uint16). Default is None to disable scaling.
            data_root_dir: if not empty, it specifies the beginning parts of the input file's
                absolute path. it's used to compute `input_file_rel_path`, the relative path to the file from
                `data_root_dir` to preserve folder structure when saving in case there are files in different
                folders with the same file names. for example:
                input_file_name: /foo/bar/test1/image.png,
                postfix: seg
                output_ext: png
                output_dir: /output,
                data_root_dir: /foo/bar,
                output will be: /output/test1/image/image_seg.png
            separate_folder: whether to save every file in a separate folder, for example: if input filename is
                `image.png`, postfix is `seg` and folder_path is `output`, if `True`, save as:
                `output/image/image_seg.png`, if `False`, save as `output/image_seg.nii`. default to `True`.
            print_log: whether to print log about the saved PNG file path, etc. default to `True`.

        """
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.output_ext = output_ext
        self.resample = resample
        self.mode: InterpolateMode = look_up_option(mode, InterpolateMode)
        self.scale = scale
        self.data_root_dir = data_root_dir
        self.separate_folder = separate_folder
        self.print_log = print_log

        self._data_index = 0

    def save(self, data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
        """
        Save data into a png file.
        The meta_data could optionally have the following keys:

            - ``'filename_or_obj'`` -- for output file name creation, corresponding to filename or object.
            - ``'spatial_shape'`` -- for data output shape.
            - ``'patch_index'`` -- if the data is a patch of big image, append the patch index to filename.

        If meta_data is None, use the default index (starting from 0) as the filename.

        Args:
            data: target data content that to be saved as a png format file.
                Assuming the data shape are spatial dimensions.
                Shape of the spatial dimensions (C,H,W).
                C should be 1, 3 or 4
            meta_data: the meta data information corresponding to the data.

        Raises:
            ValueError: When ``data`` channels is not one of [1, 3, 4].

        See Also
            :py:meth:`monai.data.png_writer.write_png`

        """
        filename = meta_data[Key.FILENAME_OR_OBJ] if meta_data else str(self._data_index)
        self._data_index += 1
        spatial_shape = meta_data.get("spatial_shape", None) if meta_data and self.resample else None
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

        if data.shape[0] == 1:
            data = data.squeeze(0)
        elif 2 < data.shape[0] < 5:
            data = np.moveaxis(np.asarray(data), 0, -1)
        else:
            raise ValueError(f"Unsupported number of channels: {data.shape[0]}, available options are [1, 3, 4]")

        write_png(
            np.asarray(data),
            file_name=path,
            output_spatial_shape=spatial_shape,
            mode=self.mode,
            scale=self.scale,
        )

        if self.print_log:
            print(f"file written: {path}.")

    def save_batch(self, batch_data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
        """Save a batch of data into png format files.

        Args:
            batch_data: target batch data content that save into png format.
            meta_data: every key-value in the meta_data is corresponding to a batch of data.

        """
        for i, data in enumerate(batch_data):  # save a batch of files
            self.save(data=data, meta_data={k: meta_data[k][i] for k in meta_data} if meta_data is not None else None)

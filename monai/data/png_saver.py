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

from typing import Dict, Optional, Union

import numpy as np
import torch

from monai.data.png_writer import write_png
from monai.data.utils import create_file_basename
from monai.utils import InterpolateMode


class PNGSaver:
    """
    Save the data as png file, it can support single data content or a batch of data.
    Typically, the data can be segmentation predictions, call `save` for single data
    or call `save_batch` to save a batch of data together. If no meta data provided,
    use index from 0 as the filename prefix.
    """

    def __init__(
        self,
        output_dir: str = "./",
        output_postfix: str = "seg",
        output_ext: str = ".png",
        resample: bool = True,
        mode: Union[InterpolateMode, str] = InterpolateMode.NEAREST,
        scale: Optional[int] = None,
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

        """
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.output_ext = output_ext
        self.resample = resample
        self.mode: InterpolateMode = InterpolateMode(mode)
        self.scale = scale

        self._data_index = 0

    def save(self, data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
        """
        Save data into a png file.
        The meta_data could optionally have the following keys:

            - ``'filename_or_obj'`` -- for output file name creation, corresponding to filename or object.
            - ``'spatial_shape'`` -- for data output shape.

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
        filename = meta_data["filename_or_obj"] if meta_data else str(self._data_index)
        self._data_index += 1
        spatial_shape = meta_data.get("spatial_shape", None) if meta_data and self.resample else None

        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        filename = create_file_basename(self.output_postfix, filename, self.output_dir)
        filename = f"{filename}{self.output_ext}"

        if data.shape[0] == 1:
            data = data.squeeze(0)
        elif 2 < data.shape[0] < 5:
            data = np.moveaxis(data, 0, -1)
        else:
            raise ValueError(f"Unsupported number of channels: {data.shape[0]}, available options are [1, 3, 4]")

        write_png(
            data,
            file_name=filename,
            output_spatial_shape=spatial_shape,
            mode=self.mode,
            scale=self.scale,
        )

    def save_batch(self, batch_data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
        """Save a batch of data into png format files.

        Args:
            batch_data: target batch data content that save into png format.
            meta_data: every key-value in the meta_data is corresponding to a batch of data.
        """
        for i, data in enumerate(batch_data):  # save a batch of files
            self.save(data, {k: meta_data[k][i] for k in meta_data} if meta_data else None)

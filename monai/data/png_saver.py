# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import torch
import numpy as np
from monai.data.png_writer import write_png
from .utils import create_file_basename


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
        interp_order: int = 3,
        mode: str = "constant",
        cval: float = 0.0,
        scale: bool = False,
    ):
        """
        Args:
            output_dir (str): output image directory.
            output_postfix (str): a string appended to all output file names.
            output_ext (str): output file extension name.
            resample: whether to resample and resize if providing spatial_shape in the metadata.
                Defaults to True.
            interp_order (int): the order of the spline interpolation, default is InterpolationCode.SPLINE3.
                This option is used when spatial_shape is specified and different from the data shape.
                The order has to be in the range 0 - 5. Defaults to 3.
            mode (`constant|edge|symmetric|reflect|wrap`):
                The mode parameter determines how the input array is extended beyond its boundaries.
                This option is used when spatial_shape is specified and different from the data shape.
                Defaults to "constant".
            cval (scalar): Value to fill past edges of input if mode is "constant". Default is 0.0.
                This option is used when spatial_shape is specified and different from the data shape.
            scale: whether to scale data with 255 and convert to uint8 for data in range [0, 1].

        """
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.output_ext = output_ext
        self.resample = resample
        self.interp_order = interp_order
        self.mode = mode
        self.cval = cval
        self.scale = scale
        self._data_index = 0

    def save(self, data: Union[torch.Tensor, np.ndarray], meta_data=None):
        """
        Save data into a png file.
        The metadata could optionally have the following keys:

            - ``'filename_or_obj'`` -- for output file name creation, corresponding to filename or object.
            - ``'spatial_shape'`` -- for data output shape.

        If meta_data is None, use the default index from 0 to save data instead.

        args:
            data (Tensor or ndarray): target data content that to be saved as a png format file.
                Assuming the data shape are spatial dimensions.
                Shape of the spatial dimensions (C,H,W).
                C should be 1, 3 or 4
            meta_data (dict): the meta data information corresponding to the data.

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
            raise ValueError("PNG image should only have 1, 3 or 4 channels.")

        write_png(
            data,
            file_name=filename,
            output_shape=spatial_shape,
            interp_order=self.interp_order,
            mode=self.mode,
            cval=self.cval,
            scale=self.scale,
        )

    def save_batch(self, batch_data: Union[torch.Tensor, np.ndarray], meta_data=None):
        """Save a batch of data into png format files.

        args:
            batch_data (Tensor or ndarray): target batch data content that save into png format.
            meta_data (dict): every key-value in the meta_data is corresponding to a batch of data.
        """
        for i, data in enumerate(batch_data):  # save a batch of files
            self.save(data, {k: meta_data[k][i] for k in meta_data} if meta_data else None)

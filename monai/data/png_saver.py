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

import os
import torch
from monai.data.png_writer import write_png

import numpy as np


class PngSaver:
    """
    Save the data as png file, it can support single data content or a batch of data.
    Typically, the data can be segmentation predictions, call `save` for single data
    or call `save_batch` to save a batch of data together. If no meta data provided,
    use index from 0 as the filename prefix.
    """

    def __init__(
        self, output_dir="./", output_postfix="seg", output_ext=".png", interp_order=3, mode="constant", cval=0
    ):
        """
        Args:
            output_dir (str): output image directory.
            output_postfix (str): a string appended to all output file names.
            output_ext (str): output file extension name.
            interp_order (int): the order of the spline interpolation, default is 0. 
                This option is used when spatial_shape is specified and different from the data shape.
                The order has to be in the range 0 - 5.
            mode (`reflect|constant|nearest|mirror|wrap`):
                The mode parameter determines how the input array is extended beyond its boundaries.
                This option is used when spatial_shape is specified and different from the data shape.
            cval (scalar): Value to fill past edges of input if mode is "constant". Default is 0.0.
                This option is used when spatial_shape is specified and different from the data shape.

        """
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.output_ext = output_ext
        self.interp_order = interp_order
        self.mode = mode
        self.cval = cval
        self._data_index = 0

    @staticmethod
    def _create_file_basename(postfix, input_file_name, folder_path, data_root_dir=""):
        """
        Utility function to create the path to the output file based on the input
        filename (extension is added by lib level writer before writing the file)

        Args:
            postfix (str): output name's postfix
            input_file_name (str): path to the input image file
            folder_path (str): path for the output file
            data_root_dir (str): if not empty, it specifies the beginning parts of the input file's
                absolute path. This is used to compute `input_file_rel_path`, the relative path to the file from
                `data_root_dir` to preserve folder structure when saving in case there are files in different
                folders with the same file names.
        """

        # get the filename and directory
        filedir, filename = os.path.split(input_file_name)

        # jettison the extension to have just filename
        filename, ext = os.path.splitext(filename)
        while ext != "":
            filename, ext = os.path.splitext(filename)

        # use data_root_dir to find relative path to file
        filedir_rel_path = ""
        if data_root_dir:
            filedir_rel_path = os.path.relpath(filedir, data_root_dir)

        # sub-folder path will be original name without the extension
        subfolder_path = os.path.join(folder_path, filedir_rel_path, filename)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # add the sub-folder plus the postfix name to become the file basename in the output path
        return os.path.join(subfolder_path, filename + "_" + postfix)

    def save(self, data, meta_data=None):
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
        spatial_shape = meta_data.get("spatial_shape", None) if meta_data else None

        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        filename = self._create_file_basename(self.output_postfix, filename, self.output_dir)
        filename = "{}{}".format(filename, self.output_ext)

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
        )

    def save_batch(self, batch_data, meta_data=None):
        """Save a batch of data into png format files.

        args:
            batch_data (Tensor or ndarray): target batch data content that save into png format.
            meta_data (dict): every key-value in the meta_data is corresponding to a batch of data.
        """
        for i, data in enumerate(batch_data):  # save a batch of files
            self.save(data, {k: meta_data[k][i] for k in meta_data} if meta_data else None)

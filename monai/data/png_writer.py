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

import numpy as np
from skimage import io

from monai.transforms import Resize
from monai.utils.misc import ensure_tuple_rep


def write_png(
    data, file_name, output_shape=None, interp_order: str = "bicubic", scale: bool = False, plugin=None, **plugin_args,
):
    """
    Write numpy data into png files to disk.
    Spatially it supports HW for 2D.(H,W) or (H,W,3) or (H,W,4)
    It's based on skimage library: https://scikit-image.org/docs/dev/api/skimage

    Args:
        data (numpy.ndarray): input data to write to file.
        file_name (string): expected file name that saved on disk.
        output_shape (None or tuple of ints): output image shape.
        interp_order (`nearest|linear|bilinear|bicubic|trilinear|area`):
            the interpolation mode. Default="bicubic".
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        scale (bool): whether to postprocess data by clipping to [0, 1] and scaling [0, 255] (uint8).
        plugin (string): name of plugin to use in `imsave`. By default, the different plugins
            are tried(starting with imageio) until a suitable candidate is found.
        plugin_args (keywords): arguments passed to the given plugin.

    """
    assert isinstance(data, np.ndarray), "input data must be numpy array."

    if output_shape is not None:
        output_shape = ensure_tuple_rep(output_shape, 2)
        xform = Resize(spatial_size=output_shape, interp_order=interp_order)
        _min, _max = np.min(data), np.max(data)
        if len(data.shape) == 3:
            data = np.moveaxis(data, -1, 0)  # to channel first
            data = xform(data)
            data = np.moveaxis(data, 0, -1)
        else:  # (H, W)
            data = np.expand_dims(data, 0)  # make a channel
            data = xform(data)[0]  # first channel
        if interp_order != "nearest":
            data = np.clip(data, _min, _max)

    if scale:
        data = np.clip(data, 0.0, 1.0)  # png writer only can scale data in range [0, 1].
        data = 255 * data
    data = data.astype(np.uint8)
    io.imsave(file_name, data, plugin=plugin, **plugin_args)
    return

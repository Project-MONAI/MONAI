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

from typing import Optional, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.croppad.array import SpatialPad
from monai.transforms.utils import rescale_array
from monai.transforms.utils_pytorch_numpy_unification import repeat
from monai.utils.module import optional_import
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

plt, _ = optional_import("matplotlib", name="pyplot")
cm, _ = optional_import("matplotlib", name="cm")

__all__ = ["matshow3d", "blend_images"]


def matshow3d(
    volume,
    fig=None,
    title: Optional[str] = None,
    figsize=(10, 10),
    frames_per_row: Optional[int] = None,
    frame_dim: int = -3,
    channel_dim: Optional[int] = None,
    vmin=None,
    vmax=None,
    every_n: int = 1,
    interpolation: str = "none",
    show=False,
    fill_value=np.nan,
    margin: int = 1,
    dtype=np.float32,
    **kwargs,
):
    """
    Create a 3D volume figure as a grid of images.

    Args:
        volume: 3D volume to display. data shape can be `BCHWD`, `CHWD` or `HWD`.
            Higher dimensional arrays will be reshaped into (-1, H, W, [C]), `C` depends on `channel_dim` arg.
            A list of channel-first (C, H[, W, D]) arrays can also be passed in,
            in which case they will be displayed as a padded and stacked volume.
        fig: matplotlib figure or Axes to use. If None, a new figure will be created.
        title: title of the figure.
        figsize: size of the figure.
        frames_per_row: number of frames to display in each row. If None, sqrt(firstdim) will be used.
        frame_dim: for higher dimensional arrays, which dimension from (`-1`, `-2`, `-3`) is moved to
            the `-3` dimension. dim and reshape to (-1, H, W) shape to construct frames, default to `-3`.
        channel_dim: if not None, explicitly specify the channel dimension to be transposed to the
            last dimensionas shape (-1, H, W, C). this can be used to plot RGB color image.
            if None, the channel dimension will be flattened with `frame_dim` and `batch_dim` as shape (-1, H, W).
            note that it can only support 3D input image. default is None.
        vmin: `vmin` for the matplotlib `imshow`.
        vmax: `vmax` for the matplotlib `imshow`.
        every_n: factor to subsample the frames so that only every n-th frame is displayed.
        interpolation: interpolation to use for the matplotlib `matshow`.
        show: if True, show the figure.
        fill_value: value to use for the empty part of the grid.
        margin: margin to use for the grid.
        dtype: data type of the output stacked frames.
        kwargs: additional keyword arguments to matplotlib `matshow` and `imshow`.

    See Also:
        - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
        - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.matshow.html

    Example:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from monai.visualize import matshow3d
        # create a figure of a 3D volume
        >>> volume = np.random.rand(10, 10, 10)
        >>> fig = plt.figure()
        >>> matshow3d(volume, fig=fig, title="3D Volume")
        >>> plt.show()
        # create a figure of a list of channel-first 3D volumes
        >>> volumes = [np.random.rand(1, 10, 10, 10), np.random.rand(1, 10, 10, 10)]
        >>> fig = plt.figure()
        >>> matshow3d(volumes, fig=fig, title="List of Volumes")
        >>> plt.show()

    """
    vol = convert_data_type(data=volume, output_type=np.ndarray)[0]
    if channel_dim is not None:
        if channel_dim not in [0, 1] or vol.shape[channel_dim] not in [1, 3, 4]:
            raise ValueError("channel_dim must be: None, 0 or 1, and channels of image must be 1, 3 or 4.")

    if isinstance(vol, (list, tuple)):
        # a sequence of channel-first volumes
        if not isinstance(vol[0], np.ndarray):
            raise ValueError("volume must be a list of arrays.")
        pad_size = np.max(np.asarray([v.shape for v in vol]), axis=0)
        pad = SpatialPad(pad_size[1:])  # assuming channel-first for item in vol
        vol = np.concatenate([pad(v) for v in vol], axis=0)
    else:  # ndarray
        while len(vol.shape) < 3:
            vol = np.expand_dims(vol, 0)  # type: ignore  # so that we display 2d as well

    if channel_dim is not None:  # move the expected dim to construct frames with `B` dim
        vol = np.moveaxis(vol, frame_dim, -4)  # type: ignore
        vol = vol.reshape((-1, vol.shape[-3], vol.shape[-2], vol.shape[-1]))
    else:
        vol = np.moveaxis(vol, frame_dim, -3)  # type: ignore
        vol = vol.reshape((-1, vol.shape[-2], vol.shape[-1]))
    vmin = np.nanmin(vol) if vmin is None else vmin
    vmax = np.nanmax(vol) if vmax is None else vmax

    # subsample every_n-th frame of the 3D volume
    vol = vol[:: max(every_n, 1)]
    if not frames_per_row:
        frames_per_row = int(np.ceil(np.sqrt(len(vol))))
    # create the grid of frames
    cols = max(min(len(vol), frames_per_row), 1)
    rows = int(np.ceil(len(vol) / cols))
    width = [[0, cols * rows - len(vol)]]
    if channel_dim is not None:
        width += [[0, 0]]  # add pad width for the channel dim
    width += [[margin, margin]] * 2
    vol = np.pad(vol.astype(dtype, copy=False), width, mode="constant", constant_values=fill_value)  # type: ignore
    im = np.block([[vol[i * cols + j] for j in range(cols)] for i in range(rows)])
    if channel_dim is not None:
        # move channel dim to the end
        im = np.moveaxis(im, 0, -1)

    # figure related configurations
    if isinstance(fig, plt.Axes):
        ax = fig
    else:
        if fig is None:
            fig = plt.figure(tight_layout=True)
        if not fig.axes:
            fig.add_subplot(111)
        ax = fig.axes[0]
    ax.matshow(im, vmin=vmin, vmax=vmax, interpolation=interpolation, **kwargs)
    ax.axis("off")

    if title is not None:
        ax.set_title(title)
    if figsize is not None and hasattr(fig, "set_size_inches"):
        fig.set_size_inches(figsize)
    if show:
        plt.show()
    return fig, im


def blend_images(
    image: NdarrayOrTensor,
    label: NdarrayOrTensor,
    alpha: Union[float, NdarrayOrTensor] = 0.5,
    cmap: str = "hsv",
    rescale_arrays: bool = True,
    transparent_background: bool = True,
):
    """
    Blend an image and a label. Both should have the shape CHW[D].
    The image may have C==1 or 3 channels (greyscale or RGB).
    The label is expected to have C==1.

    Args:
        image: the input image to blend with label data.
        label: the input label to blend with image data.
        alpha: this specifies the weighting given to the label, where 0 is completely
            transparent and 1 is completely opaque. This can be given as either a
            single value or an array/tensor that is the same size as the input image.
        cmap: specify colormap in the matplotlib, default to `hsv`, for more details, please refer to:
            https://matplotlib.org/2.0.2/users/colormaps.html.
        rescale_arrays: whether to rescale the array to [0, 1] first, default to `True`.
        transparent_background: if true, any zeros in the label field will not be colored.

    .. image:: ../../docs/images/blend_images.png

    """

    if label.shape[0] != 1:
        raise ValueError("Label should have 1 channel.")
    if image.shape[0] not in (1, 3):
        raise ValueError("Image should have 1 or 3 channels.")
    if image.shape[1:] != label.shape[1:]:
        raise ValueError("image and label should have matching spatial sizes.")
    if isinstance(alpha, (np.ndarray, torch.Tensor)):
        if image.shape[1:] != alpha.shape[1:]:
            raise ValueError("if alpha is image, size should match input image and label.")

    # rescale arrays to [0, 1] if desired
    if rescale_arrays:
        image = rescale_array(image)
        label = rescale_array(label)
    # convert image to rgb (if necessary) and then rgba
    if image.shape[0] == 1:
        image = repeat(image, 3, axis=0)

    def get_label_rgb(cmap: str, label: NdarrayOrTensor):
        _cmap = cm.get_cmap(cmap)
        label_np, *_ = convert_data_type(label, np.ndarray)
        label_rgb_np = _cmap(label_np[0])
        label_rgb_np = np.moveaxis(label_rgb_np, -1, 0)[:3]
        label_rgb, *_ = convert_to_dst_type(label_rgb_np, label)
        return label_rgb

    label_rgb = get_label_rgb(cmap, label)
    if isinstance(alpha, (torch.Tensor, np.ndarray)):
        w_label = alpha
    elif isinstance(label, torch.Tensor):
        w_label = torch.full_like(label, alpha)
    else:
        w_label = np.full_like(label, alpha)
    if transparent_background:
        # where label == 0 (background), set label alpha to 0
        w_label[label == 0] = 0
    w_image = 1 - w_label
    return w_image * image + w_label * label_rgb

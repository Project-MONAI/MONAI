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

from typing import Optional

import numpy as np

from monai.transforms.croppad.array import SpatialPad
from monai.utils.module import optional_import
from monai.utils.type_conversion import convert_data_type

plt, _ = optional_import("matplotlib", name="pyplot")

__all__ = ["matshow3d"]


def matshow3d(
    volume,
    fig=None,
    title: Optional[str] = None,
    figsize=(10, 10),
    frames_per_row: Optional[int] = None,
    vmin=None,
    vmax=None,
    every_n: int = 1,
    interpolation: str = "none",
    show=False,
    fill_value=np.nan,
    **kwargs,
):
    """
    Create a 3D volume figure as a grid of images.

    Args:
        volume: 3D volume to display. Higher dimensional arrays will be reshaped into (-1, H, W).
            A list of channel-first (C, H[, W, D]) images can also be passed in,
            in which case they will be displayed as a padded and stacked volume.
        fig: matplotlib figure to use. If None, a new figure will be created.
        title: Title of the figure.
        figsize: Size of the figure.
        frames_per_row: Number of frames to display in each row. If None, sqrt(firstdim) will be used.
        vmin: `vmin` for the matplotlib `imshow`.
        vmax: `vmax` for the matplotlib `imshow`.
        every_n: factor to subsample the frames so that only every n-th frame is displayed.
        interpolation: interpolation to use for the matplotlib `matshow`.
        show: if True, show the figure.
        fill_value: value to use for the empty part of the grid.
        kwargs: additional keyword arguments to matplotlib `matshow` and `imshow`.

    See Also:
        - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
        - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.matshow.html

    Example:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from monai.visualize import matshow3d
        # figure 3D volume
        >>> volume = np.random.rand(10, 10, 10)
        >>> fig = plt.figure()
        >>> matshow3d(volume, fig=fig, title="3D Volume")
        >>> plt.show()
        # figure a list of 3D volumes
        >>> volumes = [np.random.rand(1, 10, 10, 10), np.random.rand(1, 10, 10, 10)]
        >>> fig = plt.figure()
        >>> matshow3d(volumes, fig=fig, title="List of Volumes")
        >>> plt.show()

    """
    vol: np.ndarray = convert_data_type(data=volume, output_type=np.ndarray)[0]  # type: ignore
    if isinstance(vol, (list, tuple)):
        # a sequence of channel-first volumes
        if not isinstance(vol[0], np.ndarray):
            raise ValueError("volume must be a array of a list of arrays.")
        pad_size = np.max(np.asarray([v.shape for v in vol]), axis=0)
        pad = SpatialPad(pad_size[1:])  # assuming channel-first for item in vol
        vol = np.concatenate([pad(v) for v in vol], axis=0)
    else:
        while len(vol.shape) < 3:
            vol = np.expand_dims(vol, 0)  # so that we display 1d and 2d as well
    if len(vol.shape) > 3:
        vol = vol.reshape((-1, vol.shape[-2], vol.shape[-1]))
    vmin = np.nanmin(vol) if vmin is None else vmin
    vmax = np.nanmax(vol) if vmax is None else vmax

    # making grid of every_n-th frame
    vol = vol[:: max(every_n, 1)]
    if not frames_per_row:
        frames_per_row = int(np.ceil(np.sqrt(len(vol))))
    frames_per_row = max(min(len(vol), frames_per_row), 1)
    height, width = vol.shape[-2:]
    n_rows = int(np.ceil(len(vol) / frames_per_row))
    im = np.full(shape=(height * n_rows, width * frames_per_row), fill_value=fill_value, dtype=vol.dtype)
    for i in range(n_rows):
        sub_vol = np.hstack(vol[slice(frames_per_row * i, frames_per_row * (i + 1))])
        im[height * i : height * (i + 1), : sub_vol.shape[1]] = sub_vol

    # figure related configurations
    if fig is None:
        fig = plt.figure(tight_layout=True)
    if not fig.axes:
        fig.add_subplot(111)
    ax = fig.axes[0]
    ax.matshow(im, vmin=vmin, vmax=vmax, interpolation=interpolation, **kwargs)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    if figsize is not None:
        fig.set_size_inches(figsize)
    if show:
        plt.show()
    return fig, im

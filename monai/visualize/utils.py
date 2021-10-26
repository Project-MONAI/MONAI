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

from monai.transforms import SpatialPad
from monai.utils.module import optional_import

plt, _ = optional_import("matplotlib", name="pyplot")


__all__ = ["matshow3d"]


def matshow3d(
    vol,
    fig=None,
    title: Optional[str] = None,
    figsize=(10, 10),
    frames_per_row: Optional[int] = None,
    vmin=None,
    vmax=None,
    every_n: int = 1,
    interpolation: str = "none",
    **kwargs,
):
    """
    Display a 3D volume as a grid of images.

    Args:
        vol: 3D volume to display. Higher dimentional arrays will be reshaped into (-1, H, W).
        fig: matplotlib figure to use. If None, a new figure will be created.
        title: Title of the figure.
        figsize: Size of the figure.
        frames_per_row: Number of frames to display in each row. If None, sqrt(firstdim) will be used.
        vmin: `vmin` for the matplotlib `imshow`.
        vmax: `vmax` for the matplotlib `imshow`.
        every_n: factor to subsample the frames so that only every n-th frame is displayed.
        kwargs: additional keyword arguments to matplotlib `matshow` and `imshow`.

    See Also:
        - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
        - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.matshow.html
    """
    if isinstance(vol, (list, tuple)):
        # a sequence of channel-first volumes
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
    vol = vol[:: max(every_n, 1)]
    if not frames_per_row:
        frames_per_row = int(np.ceil(np.sqrt(len(vol))))
    frames_per_row = max(min(len(vol), frames_per_row), 0)

    im = np.hstack(vol[0:frames_per_row])
    height, width = im.shape[-2:]
    for i in range(int(np.ceil(len(vol) / frames_per_row))):
        sub_vol = vol[frames_per_row * i : frames_per_row * (i + 1)]
        if sub_vol.shape[0] == 0:
            break
        sub_vol = np.hstack(sub_vol)
        missing = width - sub_vol.shape[1]
        if missing:  # pad the image with np.nan
            sub_vol = np.hstack([sub_vol, np.nan * np.ones((height, missing))])
        im = np.concatenate([im, sub_vol], 0)

    if fig is None:
        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[0]
    ax.matshow(im, vmin=vmin, vmax=vmax, interpolation=interpolation, **kwargs)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    if figsize is not None:
        fig.set_size_inches(figsize)
    return fig, im

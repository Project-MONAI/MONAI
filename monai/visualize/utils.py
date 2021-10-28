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

# from monai.transforms import (
#     AddChanneld,
#     Compose,
#     LoadImaged,
#     Orientationd,
#     Pad,
#     RandSpatialCropd,
#     RandSpatialCropSamplesd,
#     Rotate90d,
#     ScaleIntensityd,
#     SpatialPad,
# )

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
        volume: 3D volume to display. Higher dimentional arrays will be reshaped into (-1, H, W).
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
    if show:
        plt.show()
    return fig, im


# if __name__ == "__main__":
#     keys = ("image",)
#     transforms = Compose(
#         [
#             LoadImaged(keys),
#             AddChanneld(keys),
#             ScaleIntensityd(keys),
#             Rotate90d(keys, spatial_axes=(0, 2)),
#             # RandSpatialCropSamplesd("image", roi_size=(32, 32, 20), random_size=False, num_samples=10),
#         ]
#     )
#     # image_path = "/Users/wenqili/Documents/MONAI/MarsAtlas-MNI-Colin27/colin27_MNI.nii"
#     image_path = "/Users/wenqili/Documents/MONAI/MarsAtlas-MNI-Colin27/colin27_MNI_MarsAtlas.nii"
#     ims = transforms({"image": image_path})
#     out = matshow3d((ims["image"][0],), every_n=4, frames_per_row=10, show=True)[0]
#     # out = matshow3d([im["image"] for im in ims], every_n=2, frames_per_row=10)[0]

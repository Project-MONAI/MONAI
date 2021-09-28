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

import os
import pathlib
import tempfile
from copy import deepcopy
from glob import glob
from typing import TYPE_CHECKING

import numpy as np

from monai.apps import download_and_extract
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    MapTransform,
    RandFlip,
    RandFlipd,
    Randomizable,
    Rotate90d,
    ScaleIntensityd,
    SpatialPadd,
)
from monai.utils.enums import CommonKeys
from monai.utils.module import optional_import

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    has_matplotlib = True

else:
    plt, has_matplotlib = optional_import("matplotlib.pyplot")


KEYS = [CommonKeys.IMAGE, CommonKeys.LABEL]


def get_data():

    cache_dir = os.environ.get("MONAI_DATA_DIRECTORY") or tempfile.mkdtemp()
    fname = "MarsAtlas-MNI-Colin27.zip"
    url = "https://www.dropbox.com/s/ndz8qtqblkciole/" + fname + "?dl=1"
    out_path = os.path.join(cache_dir, "MarsAtlas-MNI-Colin27")
    zip_path = os.path.join(cache_dir, fname)

    download_and_extract(url, zip_path, out_path)

    image, label = sorted(glob(os.path.join(out_path, "*.nii")))

    data = {CommonKeys.IMAGE: image, CommonKeys.LABEL: label}

    transforms = Compose(
        [
            LoadImaged(KEYS),
            AddChanneld(KEYS),
            ScaleIntensityd(CommonKeys.IMAGE),
            Rotate90d(KEYS, spatial_axes=[0, 2]),
        ]
    )
    data = transforms(data)
    im = data[CommonKeys.IMAGE]
    max_size = max(im.shape)
    data = SpatialPadd(KEYS, (max_size, max_size, max_size))(data)
    return {k: data[k] for k in KEYS}


def update_docstring(code_path, relative_out_file, transform_name):
    with open(code_path) as f:
        contents = f.readlines()
    doc_start = None
    for i, line in enumerate(contents):
        # find the line containing start of the transform documentation
        if "`" + transform_name + "`" in line:
            doc_start = i
            break
    if doc_start is None:
        raise RuntimeError("Couldn't find transform documentation")

    # if image is already in docs, nothing to do
    image_line = doc_start + 2
    if ".. image" in contents[image_line]:
        return

    contents_orig = deepcopy(contents)
    contents.insert(image_line, ".. image:: " + relative_out_file + "\n")
    contents.insert(image_line + 1, "    :alt: example of " + transform_name + "\n")

    assert len(contents) == len(contents_orig) + 2

    with open(code_path, "w") as f:
        f.writelines(contents)


def pre_process_data(data, ndim, is_map):
    if ndim == 2:
        for k in KEYS:
            data[k] = data[k][..., data[k].shape[-1] // 2]

    return data if is_map else data[CommonKeys.IMAGE]


def remove_channel(image, label, is_map):
    image = image[0]
    if is_map:
        label = label[0]
    return image, label


def get_2d_slice(image, view):
    shape = image.shape
    slices = [slice(0, s) for s in shape]
    _slice = shape[view] // 2
    slices[view] = slice(_slice, _slice + 1)
    slices = tuple(slices)
    return np.squeeze(image[slices], view)


def get_stacked_2d_ims(im):
    return np.hstack([get_2d_slice(im, view) for view in range(3)])


def get_stacked_before_after(before, after):
    return np.vstack([get_stacked_2d_ims(d[0]) for d in (before, after)])


def save_image(images, labels, filename):
    sizes = images.shape
    fig = plt.figure()
    fig.set_size_inches(1.0 * sizes[1] / sizes[0], 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(images, cmap="gray")
    if labels is not None:
        ax.imshow(labels, cmap="hsv", alpha=0.9)
    fig.savefig(filename, dpi=images.shape[0])
    plt.close(fig)


def create_transform_im(transform, data, ndim, seed=0):

    if not has_matplotlib:
        raise RuntimeError

    if isinstance(transform, Randomizable):
        transform.set_random_state(seed)

    # set output folder and create if necessary
    docs_dir = pathlib.Path(__file__).parent.parent.parent
    docs_dir = os.path.join(docs_dir, "docs", "source")
    relative_im_dir = os.path.join("..", "images", "transforms")
    im_dir = os.path.join(docs_dir, relative_im_dir)
    os.makedirs(im_dir, exist_ok=True)

    # Path is transform name
    transform_name = transform.__class__.__name__
    out_fname = transform_name + ".png"
    out_file = os.path.join(im_dir, out_fname)
    relative_out_file = os.path.join(relative_im_dir, out_fname)

    is_map = isinstance(transform, MapTransform)
    data_in = pre_process_data(data, ndim, is_map)

    data_tr = transform(data_in)

    if ndim != 3:
        raise NotImplementedError

    image_before = data_in[CommonKeys.IMAGE] if is_map else data_in
    image_after = data_tr[CommonKeys.IMAGE] if is_map else data_tr
    stacked_images = get_stacked_before_after(image_before, image_after)
    stacked_labels = None
    if is_map:
        label_before = data_in[CommonKeys.LABEL]
        label_after = data_tr[CommonKeys.LABEL]
        stacked_labels = get_stacked_before_after(label_before, label_after)
        stacked_labels[stacked_labels == 0] = np.nan

    save_image(stacked_images, stacked_labels, out_file)

    rst_path = os.path.join(docs_dir, "transforms.rst")
    update_docstring(rst_path, relative_out_file, transform_name)


if __name__ == "__main__":
    data = get_data()
    create_transform_im(RandFlip(prob=1, spatial_axis=2), data, 3)
    create_transform_im(RandFlipd(KEYS, prob=1, spatial_axis=2), data, 3)

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

from monai.apps.datasets import DecathlonDataset
import os
import pathlib
from monai.transforms import (
    RandFlip,
    LoadImaged,
    Compose,
    Lambdad,
    MapTransform,
    EnsureChannelFirstd,
    AddChanneld,
)
import matplotlib.pyplot as plt
from copy import deepcopy
from monai.utils.enums import CommonKeys

KEYS = [CommonKeys.IMAGE, CommonKeys.LABEL]

def get_data(ndim, is_map):

    cache_dir = os.environ.get("MONAI_DATA_DIRECTORY")
    if cache_dir is None:
        raise RuntimeError("Requires `MONAI_DATA_DIRECTORY` to be set")

    brats_dataset = DecathlonDataset(
        root_dir=cache_dir,
        task="Task01_BrainTumour",
        transform=None,
        section="training",
        download=True,
        num_workers=4,
        cache_num=0,
    )

    transforms = Compose([
        LoadImaged(KEYS),
        EnsureChannelFirstd(CommonKeys.IMAGE),
        Lambdad(CommonKeys.IMAGE, lambda x: x[0][None]),
        AddChanneld(CommonKeys.LABEL)
    ])
    data_full = transforms(brats_dataset[1])

    data = {}
    for k in KEYS:
        data[k] = data_full[k]
        if ndim == 2:
            data[k] = data[k][..., data[k].shape[-1]//2]

    return data if is_map else data[CommonKeys.IMAGE]

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


def create_transform_gif(transform, ndim):
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
    data_in = get_data(ndim, is_map)

    data_tr = transform(data_in)

    fig, axs = plt.subplots(1, 2)
    for data, ax in zip((data_in, data_tr), axs):

        # remove channel component and if 3d, take mid-slice of last dim
        if is_map:
            for k in KEYS:
                data[k] = data[k][0]
                if ndim == 3:
                    data[k] = data[k][..., data[k].shape[-1]//2]
        else:
            data = data[0]
            if ndim == 3:
                data = data[..., data.shape[-1]//2]
        if is_map:
            im_show = ax.imshow(data[KEYS[0]])
        else:
            im_show = ax.imshow(data)
        ax.axis("off")
        fig.colorbar(im_show, ax=ax)
    plt.savefig(out_file)
    plt.close(fig)

    rst_path = os.path.join(docs_dir, "transforms.rst")
    update_docstring(rst_path, relative_out_file, transform_name)

if __name__ == "__main__":
    create_transform_gif(RandFlip(prob=1, spatial_axis=1), 3)
    # create_transform_gif(RandFlipd(KEYS, prob=1, spatial_axis=1), 3)
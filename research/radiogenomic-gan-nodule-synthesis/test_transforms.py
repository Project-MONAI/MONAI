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

import matplotlib.pyplot as plt
import numpy as np
import torch
from RGDataPreprocessor import load_rg_data
from RGDataPreprocessorOld import load_rg_data as load_rg_data_old
from RGDataset import RGDataset

from monai.data import ITKReader
from monai.transforms import (
    AddChanneld,
    Compose,
    DataStatsd,
    LoadImaged,
    LoadNiftid,
    NormalizeIntensityd,
    Resized,
    ShiftIntensityd,
    SqueezeDimd,
    ThresholdIntensityd,
    ToTensord,
)
from monai.utils.enums import InterpolateMode
from monai.utils.misc import set_determinism

ds_rg = None
ds_og = None


def load_dataset():
    set_determinism(1)
    input_data_dir = "/home/gagan/code/MONAI/research/radiogenomic-gan-nodule-synthesis/data"
    print("Input Data Dir: %s" % input_data_dir)
    data_dict = load_rg_data_old(input_data_dir)

    # Define image transform pipeline.
    image_size = 128
    image_shape = [image_size, image_size]

    train_transforms = Compose(
        [
            LoadNiftid(keys=["image", "seg", "base"], dtype=np.float64),
            SqueezeDimd(keys=["embedding"], dim=0),
            AddChanneld(keys=["image", "seg"]),
            ThresholdIntensityd(keys=["seg"], threshold=0.5),
            Resized(keys=["image", "seg", "base"], spatial_size=image_shape, mode=InterpolateMode.AREA),
            ThresholdIntensityd(keys=["image", "base"], threshold=-1000, above=True, cval=-1000),
            ThresholdIntensityd(keys=["image", "base"], threshold=500, above=False, cval=500),
            ShiftIntensityd(keys=["image", "base"], offset=1000),
            NormalizeIntensityd(
                keys=["image", "base"],
                subtrahend=np.full(image_shape, 750.0),
                divisor=np.full(image_shape, 750.0),
                channel_wise=True,
            ),
            NormalizeIntensityd(
                keys=["seg"], subtrahend=np.full(image_shape, 0.5), divisor=np.full(image_shape, 0.5), channel_wise=True
            ),
            ToTensord(keys=["image", "seg", "base"]),
        ]
    )

    # Create dataset and dataloader.
    cache_max: int = 3
    dataset = RGDataset(data_dict, train_transforms, cache_max)
    return dataset


def load_ds_new():
    set_determinism(1)
    input_data_dir = "/home/gagan/code/MONAI/research/radiogenomic-gan-nodule-synthesis/data2"
    print("Input Data Dir: %s" % input_data_dir)

    data_dict = load_rg_data(input_data_dir)

    # Define image transform pipeline.
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"], reader=ITKReader(), dtype=np.float64),
            # AddChanneld(keys=["image", "seg"]),
            # ToTensord(keys=["image", "seg", "base"]),
        ]
    )

    # Create dataset and dataloader.
    print("Loading new dataset.")
    dataset = RGDataset(data_dict, train_transforms)
    return dataset


def get_img_rg(ds, index):
    datapoint = ds[index]
    img = datapoint["image"]
    img_2d = torch.squeeze(img)
    seg = datapoint["seg"]
    seg_2d = torch.squeeze(seg)
    return img_2d, seg_2d


def load():
    print("RGGAN DS, OG DS")
    return (load_dataset(), load_og())


def load_pair(index):
    img, seg = get_img_og(ds_og, index)
    rgimg, rgseg = get_img_rg(ds_rg, index)
    return ((img, rgimg), (seg, rgseg))


def plot_pair(index):
    imgs, segs = load_pair(index)
    fig, axs = plt.subplots(2, 2)
    # axs[0, 0].title('og img')
    axs[0, 0].imshow(imgs[0])
    # axs[0, 1].title('og seg')
    axs[0, 1].imshow(segs[0])
    # axs[1, 0].title('rg img')
    axs[1, 0].imshow(imgs[1])
    # axs[1, 1].title('rg seg')
    axs[1, 1].imshow(segs[1])
    plt.show()


def plot_bg(index):
    item = ds_rg[index]
    bgs = [torch.squeeze(bg) for bg in item["base"]]
    print(len(bgs))
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(bgs[0])
    axs[0, 1].imshow(bgs[1])
    axs[1, 0].imshow(bgs[2])
    axs[1, 1].imshow(bgs[3])
    plt.show()


if __name__ == "__main__":
    # ## INIT ###
    ds_new = load_ds_new()
    # ds_rg = load_dataset()
    print(ds_new[0])

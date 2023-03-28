import os

import numpy as np

import matplotlib.pyplot as plt

import monai
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from monai.data import Dataset, DataLoader
import monai.transforms.spatial.old_array as soa
import monai.transforms.spatial.array as sla
import monai.transforms.spatial.old_dictionary as sod
import monai.transforms.spatial.dictionary as sld
import monai.transforms.croppad.old_dictionary as cld

import torch


def get_img(size, dtype=torch.float32, offset=0):
    img = torch.zeros(size, dtype=dtype)
    if len(size) == 2:
        for j in range(size[0]):
            for i in range(size[1]):
                img[j, i] = i + j * size[0] + offset
    else:
        for k in range(size[0]):
            for j in range(size[1]):
                for i in range(size[2]):
                    img[k, j, i] = i + j * size[0] + k * size[0] * size[1]
    return np.expand_dims(img, 0)


def plot_datas(datas, cols=4, tight=False, size=20, axis=False, titles=None):
    # print(len(datas))
    minv = min([d.min() for d in datas])
    maxv = max([d.max() for d in datas])
    rows = len(datas) // cols if len(datas) % cols == 0 else len(datas) // cols + 1
    fig, ax = plt.subplots(rows, cols, figsize=(size, size * rows / cols))
    print(fig, ax)
    if tight == True:
        plt.tight_layout()

    if titles is not None:
        if len(titles) != len(datas):
            raise ValueError("titles must be the same length as data if set")

    for i_d, d in enumerate(datas):
        if axis == False:
            ax[i_d // cols, i_d % cols].axis('off')
        if titles is not None:
            ax[i_d // cols, i_d % cols].set_title(titles[i_d])
            ax[i_d // cols, i_d % cols].title.set_fontsize(28)
        if len(datas) <= cols:
            ax[i_d // cols, i_d % cols].imshow(d[0, ...] if len(d.shape) > 2 else d, vmin=minv, vmax=maxv)
        else:
            ax[i_d // cols, i_d % cols].imshow(d[0, ...] if len(d.shape) > 2 else d)


def mid_slice(vol):
    return vol[0, ..., vol.shape[-1] // 2]

# -----------------------------------------------------------------------------

def lazy_rotate_pipeline(flip=False, lazy=True):
    keys = ('image', 'label')
    load_imaged = LoadImaged(keys=keys, image_only=True)
    ensure_channel_firstd = EnsureChannelFirstd(keys=keys)
    flip_d = sld.Flipd(keys=keys, spatial_axis=0, lazy_evaluation=lazy)
    resize_d = sld.Resized(keys=keys, spatial_size=(128, 128, 64), mode=("bilinear", "nearest"), lazy_evaluation=lazy)
    scale_d = sld.Zoomd(keys=keys, zoom=0.75, mode=("bilinear", "nearest"), lazy_evaluation=lazy)
    rotate_d = sld.Rotated(keys=keys, angle=(0, 0, torch.pi / 4), padding_mode="border", lazy_evaluation=lazy)
    if flip:
        lazy_compose = Compose([
            load_imaged,
            ensure_channel_firstd,
            flip_d,
            resize_d,
            scale_d,
            rotate_d
        ])
    else:
        lazy_compose = Compose([
            load_imaged,
            ensure_channel_firstd,
            resize_d,
            scale_d,
            rotate_d
        ])

    return lazy_compose


def lazy_crop_pipeline(flip=False, lazy=True):
    keys = ('image', 'label')
    load_imaged = LoadImaged(keys=keys, image_only=True)
    ensure_channel_firstd = EnsureChannelFirstd(keys=keys)
    flip_d = sld.Flipd(keys=keys, spatial_axis=0, lazy_evaluation=lazy)
    resize_d = sld.Resized(keys=keys, spatial_size=(128, 128, 64), mode=("bilinear", "nearest"), lazy_evaluation=lazy)
    scale_d = sld.Zoomd(keys=keys, zoom=0.75, mode=("bilinear", "nearest"), lazy_evaluation=lazy)
    rotate_d = sld.Rotated(keys=keys, angle=(0, 0, torch.pi / 4), padding_mode="border", lazy_evaluation=lazy)
    # croppad_d = cld.CropPadd(keys=keys, slices=(slice(32, 96), slice(32, 96), slice(0, 64)), padding_mode="border",
    #                          lazy_evaluation=lazy)
    croppad_d = cld.RandCropPadd(keys=keys, sizes=(64, 64, 64), lazy_evaluation=lazy,
                                 state = np.random.RandomState(12345678))
    if flip:
        lazy_compose = Compose([
            load_imaged,
            ensure_channel_firstd,
            #             flip_d,
            resize_d,
            #             scale_d,
            #             rotate_d,
            croppad_d
        ])
    else:
        lazy_compose = Compose([
            load_imaged,
            ensure_channel_firstd,
            #             flip_d,
            resize_d,
            #             scale_d,
            #             rotate_d,
            croppad_d
        ])

    return lazy_compose

# -----------------------------------------------------------------------------

def test_end_to_end(data_desc, flip=False):
    keys = ('image', 'label')

    base_compose = Compose([
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
    ])

    l_rotate = lazy_rotate_pipeline(flip=flip, lazy=True)
    l_crop = lazy_crop_pipeline(flip=flip, lazy=True)
    n_rotate = lazy_rotate_pipeline(flip=flip, lazy=False)
    n_crop = lazy_crop_pipeline(flip=flip, lazy=False)

    base_dataset = Dataset(data_desc, transform=base_compose)
    base_dataloader = DataLoader(base_dataset, batch_size=1)

    l_rotate_dataset = Dataset(data_desc, transform=l_rotate)
    l_rotate_dataloader = DataLoader(l_rotate_dataset, batch_size=1)

    l_crop_dataset = Dataset(data_desc, transform=l_crop)
    l_crop_dataloader = DataLoader(l_crop_dataset, batch_size=1)

    n_rotate_dataset = Dataset(data_desc, transform=n_rotate)
    n_rotate_dataloader = DataLoader(n_rotate_dataset, batch_size=1)

    n_crop_dataset = Dataset(data_desc, transform=n_crop)
    n_crop_dataloader = DataLoader(n_crop_dataset, batch_size=1)

    results = [None for _ in range(12 * len(data_desc))]

    print("==base dataloader==")
    for i, src in enumerate(base_dataloader):
        ii = i * 12
        results[ii] = mid_slice(src['image'])
        results[ii + 1] = mid_slice(src['label'])
        results[ii + 6] = mid_slice(src['image'])
        results[ii + 7] = mid_slice(src['label'])

    print("==lazy rotate dataloader==")
    for i, lrte in enumerate(l_rotate_dataloader):
        ii = i * 12
        results[ii + 2] = mid_slice(lrte['image'])
        results[ii + 3] = mid_slice(lrte['label'])

    print("==lazy crop dataloader==")
    for i, lcrp in enumerate(l_crop_dataloader):
        ii = i * 12
        results[ii + 4] = mid_slice(lcrp['image'])
        results[ii + 5] = mid_slice(lcrp['label'])

    print("==nlazy rotate dataloader==")
    for i, nrte in enumerate(n_rotate_dataloader):
        ii = i * 12
        results[ii + 8] = mid_slice(nrte['image'])
        results[ii + 9] = mid_slice(nrte['label'])

    print("==nlazy crop dataloader==")
    for i, ncrp, in enumerate(n_crop_dataloader):
        ii = i * 12
        results[ii + 10] = mid_slice(ncrp['image'])
        results[ii + 11] = mid_slice(ncrp['label'])

    return results

# -----------------------------------------------------------------------------

basedir = "/home/ben/data/preprocessed/Task01_BrainTumour/orig"
template = "BRATS_{}_{}.nii.gz"
ids = ['001', '002', '003', '004']
data_srcs = [{'image': os.path.join(basedir, template.format(_id, 'image')),
              'label': os.path.join(basedir, template.format(_id, 'label'))}
             for _id in ids]

results = test_end_to_end(data_srcs, flip=True)
plot_datas(results, 6, tight=True)

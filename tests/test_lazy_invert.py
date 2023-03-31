import os
import time
import math
import numpy as np

import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
from torch.utils import data as tdata

import monai
import monai.transforms.spatial.old_array as old
import monai.transforms.spatial.old_dictionary as oldd
from monai.transforms.croppad.dictionary import RandCropPadd
from monai.transforms.lazy.functional import apply_pending
from monai.transforms.spatial.functional import spacing
# from monai.utils.mapping_stack import MetaMatrix
from monai.transforms import Invert, AddChannel, Compose, Crop, LoadImage, LoadImaged, EnsureChannelFirst, \
    EnsureChannelFirstd, Invertd, RandSpatialCropd
from monai.transforms.spatial.array import Flip, Resize, Rotate, Rotate90, Spacing, Zoom
import monai.transforms.spatial.array as lra
from monai.transforms.spatial.array import RandGridDistortion, Rand2DElastic
from monai.transforms.spatial.dictionary import Spacingd, Resized, RandFlipd, RandRotated, RandRotate90d, RandZoomd
from monai.transforms.croppad.functional import croppad
from monai.transforms.croppad.array import CropPad
from monai.data.meta_tensor import MetaTensor
from monai.losses.dice import DiceLoss
print(monai.__version__)
# !pip list | grep monai


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


def plot_datas(datas, cols=4, tight=False):
    # print(len(datas))
    minv = min([d.min() for d in datas])
    maxv = max([d.max() for d in datas])
    rows = len(datas) // cols if len(datas) % cols == 0 else len(datas) // cols + 1
    fig, ax = plt.subplots(rows, cols, figsize=(20, 20 * rows / cols))
    if tight == True:
        plt.tight_layout()
    for i_d, d in enumerate(datas):
        if len(datas) <= cols:
            ax[i_d].imshow(d[0, ...] if len(d.shape) > 2 else d, vmin=minv, vmax=maxv)
        else:
            ax[i_d // cols, i_d % cols].imshow(d[0, ...] if len(d.shape) > 2 else d)


def rand_seed(rng):
    value = rng.randint(np.int32((1 << 31) - 1), dtype=np.int32)
    #     print(value, type(value))
    return value


class RNGWrapper(np.random.RandomState):

    def __init__(self, tag, rng):
        self.tag = tag
        self.rng = rng
        self.calls = 0

    def rand(self, *args, **kwargs):
        self.calls += 1
        value = self.rng.rand(*args, **kwargs)
        print(self.tag, self.calls, value)
        return value

    def randint(self, *args, **kwargs):
        self.calls += 1
        value = self.rng.randint(*args, **kwargs)
        print(self.tag, self.calls, value)
        return value


def find_mid_label_z(label):
    first_z = None
    last_z = None
    for z in range(label.shape[-1]):
        count = np.count_nonzero(label[..., z])
        if count > 0:
            if first_z is None:
                first_z = z
            last_z = z
    return first_z, last_z, int((first_z + last_z) / 2)


def trad_pipeline():
    keys = ('image', 'label')
    masterrng = np.random.RandomState(12345678)

    resized = oldd.Resized(keys=keys, spatial_size=(192, 192, 72), mode=("area", "nearest"))
    randflipd = oldd.RandFlipd(keys=keys, prob=0.5, spatial_axis=[1, 2])
    randflipd.set_random_state(state=RNGWrapper("l", np.random.RandomState(rand_seed(masterrng))))
    rotate90d = oldd.RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1))
    rotate90d.set_random_state(state=np.random.RandomState(rand_seed(masterrng)))
    zoomd = oldd.RandZoomd(keys=keys, prob=0.5, min_zoom=0.75, max_zoom=1.25, mode=("area", "nearest"), keep_size=True)
    zoomd.set_random_state(state=np.random.RandomState(rand_seed(masterrng)))
    rotated = oldd.RandRotated(keys=keys, prob=1.0, range_z=(-torch.pi / 4, torch.pi / 4), mode=("bilinear", "nearest"),
                               align_corners=True)
    rotated.set_random_state(state=np.random.RandomState(rand_seed(masterrng)))
    pipeline = Compose([resized, randflipd, rotate90d, zoomd, rotated])
    # pipeline = Compose([randflipd, rotate90d, zoomd, rotated])

    return pipeline


def trad_pipeline_label_only():

    masterrng = np.random.RandomState(12345678)
    loadimage = LoadImage(image_only=True)
    ensurech = EnsureChannelFirst()
    resize = old.Resize(spatial_size=(192, 192, 72), mode="nearest")
    randflip = old.RandFlip(prob=0.5, spatial_axis=[1, 2])
    randflip.set_random_state(state=RNGWrapper("t", np.random.RandomState(rand_seed(masterrng))))
    rotate90 = old.RandRotate90(prob=0.5, spatial_axes=(0, 1))
    rotate90.set_random_state(state=np.random.RandomState(rand_seed(masterrng)))
    zoom = old.RandZoom(prob=1.0, min_zoom=0.75, max_zoom=1.25, mode="nearest", keep_size=True)
    zoom.set_random_state(state=np.random.RandomState(rand_seed(masterrng)))
    rotate = old.RandRotate(prob=1.0, range_z=(-torch.pi/4, torch.pi/4), mode="nearest", align_corners=True)
    rotate.set_random_state(state=np.random.RandomState(rand_seed(masterrng)))
    pipeline = Compose([loadimage, ensurech, resize, randflip, rotate90, zoom, rotate])

    return pipeline


def lazy_pipeline(lazy=True):
    keys = ('image', 'label')
    masterrng = np.random.RandomState(12345678)

    pipeline = Compose([
        # LoadImaged(keys=keys, image_only=True),
        # NibableReader
        ##EnsureChannelFirstd(keys=keys),
        # Orientation("RPS"),
        Resized(keys=keys, spatial_size=(192, 192, 72), mode=("bilinear", "nearest"), lazy_evaluation=lazy),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=[1, 2], lazy_evaluation=lazy,
                  state=RNGWrapper(np.random.RandomState(rand_seed(masterrng)))),
        RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1), lazy_evaluation=lazy,
                      state=np.random.RandomState(rand_seed(masterrng))),
        RandZoomd(keys=keys, prob=0.5, min_zoom=0.75, max_zoom=1.25, mode=("bilinear", "nearest"), keep_size=True,
                  lazy_evaluation=lazy,
                  state=np.random.RandomState(rand_seed(masterrng))),
        RandRotated(keys=keys, prob=1.0, range_z=(-torch.pi / 4, torch.pi / 4), mode=("bilinear", "nearest"),
                    align_corners=True, lazy_evaluation=lazy,
                    state=np.random.RandomState(rand_seed(masterrng))),
    ])

    return pipeline


def trad_pipeline_patch_last():

    keys = ('image', 'label')
    masterrng = np.random.RandomState(12345678)
    randfliprng = np.random.RandomState(rand_seed(masterrng))
    rotate90rng = np.random.RandomState(rand_seed(masterrng))
    zoomrng = np.random.RandomState(rand_seed(masterrng))
    rotaterng = np.random.RandomState(rand_seed(masterrng))
    patch_seed = rand_seed(masterrng)
    print("lazy patch seed:", patch_seed)
    patchrng =  np.random.RandomState(patch_seed)

    resized = oldd.Spacingd(keys=keys, pixdim=(1.0, 1.0, 155/72), mode=("bilinear", "nearest"))
    randflipd = oldd.RandFlipd(keys=keys, prob=0.5, spatial_axis=[1, 2])
    randflipd.set_random_state(state=randfliprng)
    rotate90d = oldd.RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1))
    rotate90d.set_random_state(state=rotate90rng)
    zoomd = oldd.RandZoomd(keys=keys, prob=1.0, min_zoom=0.75, max_zoom=1.25, mode=("area", "nearest"), keep_size=True)
    zoomd.set_random_state(state=zoomrng)
    rotated = oldd.RandRotated(keys=keys, prob=1.0, range_z=(-torch.pi/4, torch.pi/4), mode=("bilinear", "nearest"), align_corners=True)
    rotated.set_random_state(state=rotaterng)
    patchd = RandSpatialCropd(keys=keys, roi_size=(160, 160, 72), random_size=False)
    patchd.set_random_state(state=patchrng)
    pipeline = Compose([resized, randflipd, rotate90d, zoomd, rotated, patchd])
    # pipeline = Compose([randflipd, rotate90d, zoomd, rotated])

    return pipeline


def lazy_pipeline_patch_last(lazy=True):
    keys = ('image', 'label')
    masterrng = np.random.RandomState(12345678)
    randfliprng = np.random.RandomState(rand_seed(masterrng))
    rotate90rng = np.random.RandomState(rand_seed(masterrng))
    zoomrng = np.random.RandomState(rand_seed(masterrng))
    rotaterng = np.random.RandomState(rand_seed(masterrng))
    patch_seed = rand_seed(masterrng)
    print("lazy patch seed:", patch_seed)
    patchrng = np.random.RandomState(patch_seed)

    pipeline = Compose([
        # LoadImaged(keys=keys, image_only=True),
        # NibableReader
        ##EnsureChannelFirstd(keys=keys),
        # Orientation("RPS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 155 / 72), mode=("bilinear", "nearest"), lazy_evaluation=lazy),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=[1, 2], lazy_evaluation=lazy, state=randfliprng),
        RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1), lazy_evaluation=lazy, state=rotate90rng),
        RandZoomd(keys=keys, prob=1.0, min_zoom=0.75, max_zoom=1.25, mode=("bilinear", "nearest"), keep_size=True,
                  lazy_evaluation=lazy, state=zoomrng),
        RandRotated(keys=keys, prob=1.0, range_z=(-torch.pi / 4, torch.pi / 4), mode=("bilinear", "nearest"),
                    align_corners=True, lazy_evaluation=lazy, state=rotaterng),
        RandCropPadd(keys=keys, sizes=(160, 160, 72), lazy_evaluation=lazy, state=patchrng),
    ])

    return pipeline


def lazy_pipeline_label_only(lazy=True):
    masterrng = np.random.RandomState(12345678)

    pipeline = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(spatial_size=(192, 192, 72), mode="nearest", lazy_evaluation=lazy),
        lra.RandFlip(prob=0.5, spatial_axis=[1, 2], lazy_evaluation=lazy,
                 state=RNGWrapper("l", np.random.RandomState(rand_seed(masterrng)))),
        lra.RandRotate90(prob=0.5, spatial_axes=(0, 1), lazy_evaluation=lazy,
                     state=np.random.RandomState(rand_seed(masterrng))),
        lra.RandZoom(prob=1.0, min_zoom=0.75, max_zoom=1.25, mode="nearest", keep_size=True, lazy_evaluation=lazy,
                 state=np.random.RandomState(rand_seed(masterrng))),
        lra.RandRotate(prob=1.0, range_z=(-torch.pi / 4, torch.pi / 4), mode="nearest", align_corners=True,
                   lazy_evaluation=lazy,
                   state=np.random.RandomState(rand_seed(masterrng))),
    ])

    return pipeline


def test_invert():
    import sys
    from copy import deepcopy
    from monai.utils import set_determinism
    # from tests.utils import assert_allclose, make_nifti_image
    from monai.data import DataLoader, Dataset, MetaTensor, create_test_image_3d, decollate_batch
    from monai.transforms import (
        CastToType,
        Compose,
        EnsureChannelFirst,
        Invert,
        LoadImage,
        Orientation,
        RandAffine,
        RandAxisFlip,
        RandFlip,
        RandRotate,
        RandRotate90,
        RandZoom,
        ResizeWithPadOrCrop,
        Spacing,
    )

    num_rows = 4
    keys = ('label',)
    mode = 'nearest'
    set_determinism(seed=0)
    # im_fname = make_nifti_image(create_test_image_3d(101, 100, 107, noise_max=100)[1])  # label image, discrete
    #     data = [im_fname for _ in range(12)]

    data = ['/home/ben/data/preprocessed/Task01_BrainTumour/orig/BRATS_001_label.nii.gz',
            '/home/ben/data/preprocessed/Task01_BrainTumour/orig/BRATS_002_label.nii.gz',
            '/home/ben/data/preprocessed/Task01_BrainTumour/orig/BRATS_003_label.nii.gz',
            '/home/ben/data/preprocessed/Task01_BrainTumour/orig/BRATS_004_label.nii.gz']
    print(data)
    lazy = True
    base_images = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
        ]
    )

    #     transform_old = Compose(
    #         [
    #             LoadImage(image_only=True),
    #             EnsureChannelFirst(),
    #             # Orientation("RPS"),
    #             old.Spacing(pixdim=(1.2, 1.01, 0.9), mode=mode, dtype=np.float32),
    #             old.Flip(spatial_axis=[1, 2]),
    #             old.Rotate90(spatial_axes=(1, 2)),
    #             old.Zoom(zoom=0.75, keep_size=True),
    #             old.Rotate(angle=(np.pi, 0, 0), mode=mode, align_corners=True, dtype=np.float64),
    #             # RandAffine(prob=0.5, rotate_range=np.pi, mode="nearest"),
    #             # ResizeWithPadOrCrop(100),
    #             CastToType(dtype=torch.uint8),
    #         ]
    #     )
    #     transform_new = Compose(
    #         [
    #             LoadImage(image_only=True),
    #             EnsureChannelFirst(),
    #             # Orientation("RPS"),
    #             Spacing(pixdim=(1.2, 1.01, 0.9), mode=mode, dtype=np.float32, lazy_evaluation=lazy),
    #             Flip(spatial_axis=[1, 2], lazy_evaluation=lazy),
    #             Rotate90(spatial_axes=(1, 2), lazy_evaluation=lazy),
    #             Zoom(zoom=0.75, keep_size=True, lazy_evaluation=lazy),
    #             Rotate(angle=(0, 0, np.pi), mode=mode, align_corners=True, dtype=np.float64,
    #                    lazy_evaluation=lazy),
    #             # RandAffine(prob=0.5, rotate_range=np.pi, mode="nearest"),
    #             # ResizeWithPadOrCrop(100),
    #             CastToType(dtype=torch.uint8),
    #         ]
    #     )
    #     print(transform._forward_transforms)

    # print("loader length =", len(loader))
    # fig, ax = plt.subplots(12, 3, figsize=(12, 48))
    transform_old = trad_pipeline_label_only()
    transform_new = lazy_pipeline_label_only()

    results = [None for _ in range(num_rows * 3)]

    for i_tx, tx in enumerate([transform_old, transform_new]):
        # num workers = 0 for mac or gpu transforms
        num_workers = 0 if sys.platform != "linux" or torch.cuda.is_available() else 2
        base_dataset = Dataset(data, transform=base_images)
        base_loader = DataLoader(base_dataset, suppress_rng=True, batch_size=1)

        dataset = Dataset(data, transform=tx)
        # self.assertIsInstance(transform.inverse(dataset[0]), MetaTensor)
        loader = DataLoader(dataset, suppress_rng=True, batch_size=1)
        inverter = Invert(transform=tx, nearest_interp=True, device="cpu", post_func=torch.as_tensor)

        for i_d, d in enumerate(base_loader):
            d = decollate_batch(d)
            for item in d:
                print(item.shape)
                if i_tx == 0:
                    # results[i_d * 3] = item[0, ..., item.shape[-1] // 2]
                    results[i_d * 3] = item

        for i_d, d in enumerate(loader):
            d = decollate_batch(d)
            for item in d:
                print(np.unique(item, return_counts=True))
                orig = deepcopy(item)
                i = inverter(item)
                print(item.shape, i.shape)
                # results[i_d * 3 + i_tx + 1] = i[0, ..., i.shape[-1] // 2]
                results[i_d * 3 + i_tx + 1] = i
        # check labels match
        reverted = i.detach().cpu().numpy().astype(np.int32)
        original = LoadImage(image_only=True)(data[-1])
        n_good = np.sum(np.isclose(reverted, original.numpy(), atol=1e-3))
        reverted_name = i.meta["filename_or_obj"]
        original_name = original.meta["filename_or_obj"]
        # self.assertEqual(reverted_name, original_name)
        print("invert diff", reverted.size - n_good)
        # self.assertTrue((reverted.size - n_good) < 300000, f"diff. {reverted.size - n_good}")
        set_determinism(seed=None)

    # print(['None' if r is None else r.shape for r in results])
    dl = DiceLoss(reduction="none")
    for r in range(num_rows):
        r0 = results[r * 3]
        r0h = torch.nn.functional.one_hot(r0.long())
        r0h = torch.squeeze(r0h, 0).permute(3, 0, 1, 2)
        r1 = results[r * 3 + 1]
        r1h = torch.nn.functional.one_hot(r1.long())
        r1h = torch.squeeze(r1h, 0).permute(3, 0, 1, 2)
        print(r0h.shape, r1h.shape)
        r2 = results[r * 3 + 2]
        r2h = torch.nn.functional.one_hot(r2.long())
        r2h = torch.squeeze(r2h, 0).permute(3, 0, 1, 2)

        print(r0h.shape)
        dl1 = dl(r0h, r1h).mean(dim=(1, 2, 3))
        dl2 = dl(r0h, r2h).mean(dim=(1, 2, 3))

        print(1 - dl1, 1 - dl2)

    plot_datas([r[0, ..., find_mid_label_z(r)[2]] for r in results], 3, tight=True)


def check_patch_first_pipeline_forward():
    base_dir = '/home/ben/data/preprocessed/Task01_BrainTumour/orig'
    sample_str = 'BRATS_{}_{}.nii.gz'
    sample = '001'

    iterations = 4

    img = nib.load(os.path.join(base_dir, sample_str.format(sample, 'image')))
    lbl = nib.load(os.path.join(base_dir, sample_str.format(sample, 'label')))

    ddict = {'image': img.get_fdata(), 'label': lbl.get_fdata()}

    ddict['image'] = np.transpose(ddict['image'], axes=(3, 0, 1, 2))
    ddict['label'] = np.expand_dims(ddict['label'], axis=0)

    # print(ddict['image'].shape, ddict['label'].shape)

    tp = trad_pipeline_patch_last()
    lp = lazy_pipeline_patch_last(True)

    pre_first_z, pre_last_z, pre_mid_slice = find_mid_label_z(ddict['label'])
    print(pre_first_z, pre_last_z, pre_mid_slice)
    results = []
    t_time = 0
    l_time = 0
    for i in range(iterations):
        t_start = time.time()
        t_out = tp(ddict)
        t_time += time.time() - t_start
        print(t_out['label'].shape)
        t_first_z, t_last_z, t_mid_slice = find_mid_label_z(t_out['label'])
        print(t_first_z, t_last_z, t_mid_slice)
        l_start = time.time()
        l_out = lp(ddict)
        l_time += time.time() - l_start
        l_first_z, l_last_z, l_mid_slice = find_mid_label_z(l_out['label'])
        print(l_first_z, l_last_z, l_mid_slice)

        #     results.extend([ddict['image'][0, ..., pre_mid_slice], ddict['label'][0, ..., pre_mid_slice],
        #                     t_out['image'][0, ..., post_mid_slice], t_out['label'][0, ..., post_mid_slice],
        #                     l_out['image'][0, ..., post_mid_slice], l_out['label'][0, ..., post_mid_slice]])
        results.extend([ddict['label'][0, ..., pre_mid_slice],
                        t_out['label'][0, ..., t_mid_slice],
                        l_out['label'][0, ..., l_mid_slice]])
    print(f"trad time: {t_time}, lazy time: {l_time}")
    # plot_datas(results, 6)
    plot_datas(results, 3, tight=True)

def simple_test_fwd():
    keys = ('image', 'label')
    patchd = RandSpatialCropd(keys=keys, roi_size=(192, 192, 155), random_size=False)
    patchd.set_random_state(state=RNGWrapper(np.random.RandomState(12345678)))
    d = {'image': torch.rand((1, 240, 240, 155)), 'label': torch.rand((1, 240, 240, 155))}
    rd = patchd(d)

test_invert()

# simple_test_fwd()

# check_patch_first_pipeline_forward()

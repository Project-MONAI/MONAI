import unittest

import numpy as np

import torch

from monai.transforms import Compose
import monai.transforms.spatial.old_dictionary as oldd
from monai.transforms.spatial.dictionary import Resized, RandFlipd, RandRotate90d, RandZoomd, RandRotated


def rand_seed(rng):
    value = rng.randint(np.int32((1 << 31) - 1), dtype=np.int32)
    print(value, type(value))
    return value


def trad_pipeline():
    keys = ('image', 'label')
    masterrng = np.random.RandomState(12345678)

    resized = oldd.Resized(keys=keys, spatial_size=(192, 192, 72), mode="area")
    randflipd = oldd.RandFlipd(keys=keys, spatial_axis=[1, 2])
    randflipd.set_random_state(state=np.random.RandomState(rand_seed(masterrng)))
    rotate90d = oldd.RandRotate90d(keys=keys, spatial_axes=(1, 2))
    rotate90d.set_random_state(state=np.random.RandomState(rand_seed(masterrng)))
    # zoomd = oldd.RandZoomd(keys=keys, min_zoom=0.75, max_zoom=1.25, keep_size=True)
    # zoomd.set_random_state(state=np.random.RandomState(rand_seed(masterrng)))
    # rotated = oldd.RandRotated(keys=keys, range_z=(-torch.pi / 4, torch.pi / 4), mode="bilinear", align_corners=True,
    #                            dtype=np.float64)
    # rotated.set_random_state(state=np.random.RandomState(rand_seed(masterrng)))
    pipeline = Compose([resized, randflipd])#, rotate90d, zoomd, rotated])

    def _inner(data):
        return pipeline(data)

    return _inner


def lazy_pipeline(lazy=True):
    keys = ('image', 'label')
    masterrng = np.random.RandomState(12345678)

    pipeline = Compose([
        # LoadImaged(keys=keys, image_only=True),
        # EnsureChannelFirstd(keys=keys),
        # Orientation("RPS"),
        Resized(keys=keys, spatial_size=(192, 192, 72), mode="bilinear", lazy_evaluation=lazy),
        RandFlipd(keys=keys, spatial_axis=[1, 2], lazy_evaluation=lazy,
                  state=np.random.RandomState(rand_seed(masterrng))),
        RandRotate90d(keys=keys, spatial_axes=(1, 2), lazy_evaluation=lazy,
                      state=np.random.RandomState(rand_seed(masterrng))),
        # RandZoomd(keys=keys, min_zoom=0.75, max_zoom=1.25, keep_size=True, lazy_evaluation=lazy,
        #           state=np.random.RandomState(rand_seed(masterrng))),
        # RandRotated(keys=keys, range_z=(-torch.pi / 4, torch.pi / 4), mode="bilinear", align_corners=True,
        #             dtype=np.float64, lazy_evaluation=lazy,
        #             state=np.random.RandomState(rand_seed(masterrng))),
    ])
    # print(pipeline._forward_transforms)

    def _inner(data):
        return pipeline(data)

    return _inner


class TestApplyTransforms(unittest.TestCase):


    def test_with_compose(self):
        lazy = True
        keys = ('image',)
        tp = trad_pipeline()
        lp = lazy_pipeline(True)

        ddict = {'image': torch.rand(1, 64, 64, 32), 'label': torch.rand(1, 64, 64, 32)}

        tdict = tp(ddict)
        ldict = lp(ddict)

    def test_rand_flip_rngs(self):

        keys = ('image',)

        orng = np.random.RandomState(12345678)
        orandflipd = oldd.RandFlipd(keys=keys, spatial_axis=[1, 2])
        orandflipd.set_random_state(state=orng)

        lrng = np.random.RandomState(12345678)
        lrandflipd = RandFlipd(keys=keys, spatial_axis=[1, 2], lazy_evaluation=True,
                               state=lrng)

        ddict = {'image': torch.rand(1, 64, 64, 32)} # , 'label': torch.rand(1, 64, 64, 32)}

        for i in range(10):
            odict = orandflipd(ddict)
            ldict = lrandflipd(ddict)

    def test_rand_rotate_90_rngs(self):

        keys = ('image',)

        orng = np.random.RandomState(12345678)
        o_op = oldd.RandRotate90d(keys=keys, spatial_axes=(1, 2))
        o_op.set_random_state(state=orng)

        lrng = np.random.RandomState(12345678)
        l_op = RandRotate90d(keys=keys, spatial_axes=(1, 2), lazy_evaluation=True,
                             state=lrng)

        ddict = {'image': torch.rand(1, 64, 64, 32)} # , 'label': torch.rand(1, 64, 64, 32)}

        for i in range(10):
            odict = o_op(ddict)
            ldict = l_op(ddict)

    def test_rand_zoom_rngs(self):

        keys = ('image',)

        orng = np.random.RandomState(12345678)
        o_op = oldd.RandZoomd(keys=keys, prob=0.5, min_zoom=0.75, max_zoom=1.25, keep_size=True, spatial_axes=(1, 2))
        o_op.set_random_state(state=orng)

        lrng = np.random.RandomState(12345678)
        l_op = RandZoomd(keys=keys, prob=0.5, min_zoom=0.75, max_zoom=1.25, keep_size=True, lazy_evaluation=True,
                             state=lrng)

        ddict = {'image': torch.rand(1, 64, 64, 32)} # , 'label': torch.rand(1, 64, 64, 32)}

        for i in range(10):
            odict = o_op(ddict)
            ldict = l_op(ddict)

    def test_rand_rotate_rngs(self):

        keys = ('image',)

        orng = np.random.RandomState(12345678)
        o_op = oldd.RandRotated(keys=keys, prob=0.5, range_z=(-torch.pi / 4, torch.pi / 4), mode="bilinear",
                                align_corners=True, dtype=np.float64)
        o_op.set_random_state(state=orng)

        lrng = np.random.RandomState(12345678)
        l_op = RandRotated(keys=keys, prob=0.5, range_z=(-torch.pi / 4, torch.pi / 4), mode="bilinear",
                           align_corners=True, dtype=np.float64, lazy_evaluation=True, state=lrng)

        ddict = {'image': torch.rand(1, 64, 64, 32)} # , 'label': torch.rand(1, 64, 64, 32)}

        for i in range(10):
            odict = o_op(ddict)
            ldict = l_op(ddict)
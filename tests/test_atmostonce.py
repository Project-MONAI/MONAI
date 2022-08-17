import unittest

import numpy as np


import torch

from monai.transforms import Affined
from monai.transforms.atmostonce.apply import Applyd
from monai.transforms.atmostonce.dictionary import RotateEulerd
from monai.transforms.compose import Compose
from monai.utils.enums import GridSampleMode, GridSamplePadMode

class TestRotateEulerd(unittest.TestCase):

    def test_rotate_numpy(self):
        r = RotateEulerd(('image', 'label'), [0.0, 1.0, 0.0])

        d = {
            'image': np.zeros((1, 64, 64, 32), dtype=np.float32),
            'label': np.ones((1, 64, 64, 32), dtype=np.int8)
        }
        d = r(d)

        for k, v in d.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape)
            else:
                print(k, v)

    def test_rotate_tensor(self):
        r = RotateEulerd(('image', 'label'), [0.0, 1.0, 0.0])

        d = {
            'image': torch.zeros((1, 64, 64, 32), device="cpu", dtype=torch.float32),
            'label': torch.ones((1, 64, 64, 32), device="cpu", dtype=torch.int8)
        }
        d = r(d)

        for k, v in d.items():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                print(k, v.shape)
            else:
                print(k, v)

    def test_rotate_apply(self):
        c = Compose([
            RotateEulerd(('image', 'label'), (0.0, 3.14159265 / 2, 0.0)),
            Applyd(('image', 'label'),
                   modes=(GridSampleMode.BILINEAR, GridSampleMode.NEAREST),
                   padding_modes=(GridSamplePadMode.BORDER, GridSamplePadMode.BORDER))
        ])

        image = torch.zeros((1, 16, 16, 4), device="cpu", dtype=torch.float32)
        for y in range(image.shape[-2]):
            for z in range(image.shape[-1]):
                image[0, :, y, z] = y + z * 16
        label = torch.ones((1, 16, 16, 4), device="cpu", dtype=torch.int8)
        d = {
            'image': image,
            'label': label
        }
        # plt.imshow(d['image'][0, ..., d['image'].shape[-1]//2])
        d = c(d)
        # plt.imshow(d['image'][0, ..., d['image'].shape[-1]//2])
        print(d['image'].shape)

    def test_old_affine(self):
        c = Compose([
            Affined(('image', 'label'),
                    rotate_params=(0.0, 0.0, 3.14159265 / 2))
        ])

        d = {
            'image': torch.zeros((1, 64, 64, 32), device="cpu", dtype=torch.float32),
            'label': torch.ones((1, 64, 64, 32), device="cpu", dtype=torch.int8)
        }
        d = c(d)
        print(d['image'].shape)

import numpy as np

import matplotlib.pyplot as plt

import torch
from monai.utils import GridSampleMode, GridSamplePadMode

from monai.transforms.atmostonce.apply import Applyd
from monai.transforms.atmostonce.dictionary import Rotated
from monai.transforms import Compose



def test_rotate_tensor():
    r = Rotated(('image', 'label'), [0.0, 1.0, 0.0])

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


def test_rotate_apply():
    c = Compose([
        Rotated(('image', 'label'), (0.0, 3.14159265 / 2, 0.0)),
        Applyd(('image', 'label'),
               modes=(GridSampleMode.BILINEAR, GridSampleMode.NEAREST),
               padding_modes=(GridSamplePadMode.BORDER, GridSamplePadMode.BORDER))
    ])

    d = {
        'image': torch.zeros((1, 64, 64, 32), device="cpu", dtype=torch.float32),
        'label': torch.ones((1, 64, 64, 32), device="cpu", dtype=torch.int8)
    }
    plt.imshow(d['image'][0, ..., d['image'].shape[-1]//2])
    d = c(d)
    plt.imshow(d['image'][0, ..., d['image'].shape[-1]//2])
    print(d['image'].shape)

test_rotate_apply()

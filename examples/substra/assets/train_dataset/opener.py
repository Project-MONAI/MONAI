import os
import tempfile
from glob import glob

import nibabel as nib
import substratools as tools
import torch
from torch.utils.data import DataLoader

from monai.data import CacheDataset, create_test_image_3d, list_data_collate
from monai.transforms import (
    AsChannelFirstd,
    Compose,
    LoadNiftid,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    ToTensord,
)


class MonaiTrainOpener(tools.Opener):
    def init(self):
        self._fake_samples_directory = tempfile.mkdtemp()

    def _get_loader(self, folders):
        images = []
        segs = []
        for folder in folders:
            images += glob(os.path.join(folder, "*_im.nii.gz"))
            segs += glob(os.path.join(folder, "*_seg.nii.gz"))
        images = sorted(images, key=os.path.basename)
        segs = sorted(segs, key=os.path.basename)

        files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

        transforms = Compose(
            [
                LoadNiftid(keys=["img", "seg"]),
                AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
                ScaleIntensityd(keys="img"),
                RandCropByPosNegLabeld(
                    keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
                ),
                RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
                ToTensord(keys=["img", "seg"]),
            ]
        )

        ds = CacheDataset(data=files, transform=transforms)
        loader = DataLoader(
            ds,
            batch_size=2,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )

        return loader

    def get_X(self, folders):  # noqa: N802
        loader = self._get_loader(folders)
        return loader

    def get_y(self, folders):
        return None

    def save_predictions(self, y_pred, path):
        raise NotImplementedError

    def get_predictions(self, path):
        raise NotImplementedError

    def _generate_fake_samples(self, n_samples):
        prefix = "test"
        for i in range(n_samples):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)

            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(self._fake_samples_directory, f"{prefix}_{i}_im.nii.gz"))

            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(self._fake_samples_directory, f"{prefix}_{i}_seg.nii.gz"))

    def fake_X(self, n_samples):  # noqa: N802
        self._generate_fake_samples(n_samples=n_samples)
        return self._get_loader([self._fake_samples_directory])

    def fake_y(self, n_samples):
        return None

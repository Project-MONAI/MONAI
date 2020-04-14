import os
import tempfile
import zipfile
from glob import glob

import nibabel as nib
import numpy as np
import substratools as tools
from torch.utils.data import DataLoader

from monai.data import CacheDataset, NiftiSaver, create_test_image_3d, list_data_collate
from monai.transforms import AsChannelFirstd, Compose, LoadNiftid, ScaleIntensityd, ToTensord


class MonaiTestOpener(tools.Opener):
    def init(self):
        self._fake_samples_directory = tempfile.mkdtemp()
        self._has_generated_fake = False

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
                ToTensord(keys=["img", "seg"]),
            ]
        )

        ds = CacheDataset(data=files, transform=transforms)
        loader = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

        return loader

    def _get_X_iterator(self, loader):  # noqa: N802
        for data in loader:
            yield (data["img"], data["img_meta_dict"])

    def _get_y_iterator(self, loader):
        for data in loader:
            yield (data["seg"], data["seg_meta_dict"])

    def get_X(self, folders):  # noqa: N802
        loader = self._get_loader(folders)
        return self._get_X_iterator(loader)

    def get_y(self, folders):
        loader = self._get_loader(folders)
        return self._get_y_iterator(loader)

    def save_predictions(self, y_pred, path):
        with tempfile.TemporaryDirectory() as tmp_dir:
            saver = NiftiSaver(output_dir=tmp_dir)
            for outputs, metadata in y_pred:
                saver.save_batch(outputs, metadata)
            # join all predictions in a single zip
            with zipfile.ZipFile(path, "w") as z:
                for filepath in glob(os.path.join(tmp_dir, "*/*")):
                    z.write(filepath, arcname=os.path.relpath(filepath, tmp_dir))

    def _get_predictions_iterator(self, segs):
        files = [{"seg": seg} for seg in segs]
        transforms = Compose(
            [
                LoadNiftid(keys=["seg"]),
                AsChannelFirstd(keys=["seg"], channel_dim=-1),
                ToTensord(keys=["seg"]),
            ]
        )
        ds = CacheDataset(data=files, transform=transforms)
        loader = DataLoader(ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
        for data in loader:
            yield (data["seg"], data["seg_meta_dict"])

    def get_predictions(self, path):
        # unzip predictions
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmp_dir)
        # load predictions
        segs = sorted(glob(os.path.join(tmp_dir, "*/*_seg.nii.gz")), key=os.path.basename)
        return self._get_predictions_iterator(segs)

    def _generate_fake_samples(self, n_samples):
        prefix = "test"
        for i in range(n_samples):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)

            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(self._fake_samples_directory, f"{prefix}_{i}_im.nii.gz"))

            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(self._fake_samples_directory, f"{prefix}_{i}_seg.nii.gz"))
        self._has_generated_fake = True

    def fake_X(self, n_samples):  # noqa: N802
        if not self._has_generated_fake:
            self._generate_fake_samples(n_samples=n_samples)
        loader = self._get_loader([self._fake_samples_directory])
        return self._get_X_iterator(loader)

    def fake_y(self, n_samples):
        if not self._has_generated_fake:
            self._generate_fake_samples(n_samples=n_samples)
        loader = self._get_loader([self._fake_samples_directory])
        return self._get_X_iterator(loader)

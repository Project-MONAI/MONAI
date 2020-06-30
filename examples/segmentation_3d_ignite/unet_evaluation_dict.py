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

import os
import sys
import tempfile
import shutil
from glob import glob
import logging
import nibabel as nib
import numpy as np
import torch
from ignite.engine import Engine
from torch.utils.data import DataLoader

import monai
from monai.data import list_data_collate, create_test_image_3d
from monai.inferers import sliding_window_inference
from monai.networks.utils import predict_segmentation
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadNiftid, AsChannelFirstd, ScaleIntensityd, ToTensord
from monai.handlers import SegmentationSaver, CheckpointLoader, StatsHandler, MeanDice


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    tempdir = tempfile.mkdtemp()
    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(5):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    # define transforms for image and segmentation
    val_transforms = Compose(
        [
            LoadNiftid(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys=["img", "seg"]),
            ToTensord(keys=["img", "seg"]),
        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)

    device = torch.device("cuda:0")
    net = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    net.to(device)

    # define sliding window size and batch size for windows inference
    roi_size = (96, 96, 96)
    sw_batch_size = 4

    def _sliding_window_processor(engine, batch):
        net.eval()
        with torch.no_grad():
            val_images, val_labels = batch["img"].to(device), batch["seg"].to(device)
            seg_probs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
            return seg_probs, val_labels

    evaluator = Engine(_sliding_window_processor)

    # add evaluation metric to the evaluator engine
    MeanDice(sigmoid=True, to_onehot_y=False).attach(evaluator, "Mean_Dice")

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't need to print loss for evaluator, so just print metrics, user can also customize print functions
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
    )
    val_stats_handler.attach(evaluator)

    # convert the necessary metadata from batch data
    SegmentationSaver(
        output_dir="tempdir",
        output_ext=".nii.gz",
        output_postfix="seg",
        name="evaluator",
        batch_transform=lambda batch: batch["img_meta_dict"],
        output_transform=lambda output: predict_segmentation(output[0]),
    ).attach(evaluator)
    # the model was trained by "unet_training_dict" example
    CheckpointLoader(load_path="./runs/net_checkpoint_50.pth", load_dict={"net": net}).attach(evaluator)

    # sliding window inference for one image at every iteration
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available()
    )
    state = evaluator.run(val_loader)
    print(state)
    shutil.rmtree(tempdir)


if __name__ == "__main__":
    main()

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
from glob import glob
import logging
import nibabel as nib
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import monai
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadNiftid,
    AsChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ToTensord,
)
from monai.data import create_test_image_3d, list_data_collate


def train(gpu, args):
    # initialize the distributed training process
    rank = args.ranking * args.gpus + gpu
    dist.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=rank)

    images = sorted(glob(os.path.join(args.dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(args.dir, "seg*.nii.gz")))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadNiftid(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys=["img", "seg"]),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
            ToTensord(keys=["img", "seg"]),
        ]
    )

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # create a training data sampler
    train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=rank)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        sampler=train_sampler,
    )

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device(f"cuda:{gpu}")
    model = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    # wrap the model with DistributedDataParallel module
    model = DDP(model, device_ids=[gpu])

    # start a typical PyTorch training
    epoch_loss_values = list()
    for epoch in range(5):
        if gpu == 0:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        train_sampler.set_epoch(epoch)
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            if gpu == 0:
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        if gpu == 0:
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if gpu == 0:
        print(f"train completed, epoch losses: {epoch_loss_values}")


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    set_determinism(seed=0)

    # create 40 random image, mask paris for training
    data_dir = "./testdata"
    if not os.path.exists(data_dir):
        print(f"generating synthetic data to {data_dir} (this may take a while)")
        os.makedirs(data_dir)
        for i in range(40):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)

            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(data_dir, f"img{i:d}.nii.gz"))

            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(data_dir, f"seg{i:d}.nii.gz"))

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("-r", "--ranking", default=0, type=int, help="node ranking within all the nodes")
    parser.add_argument("-d", "--dir", default="./testdata", type=str, help="directory to create random data")
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ["MASTER_ADDR"] = "10.23.137.29"
    os.environ["MASTER_PORT"] = "8888"
    mp.spawn(train, nprocs=args.gpus, args=(args,))


# usage: python unet_training_ddp.py -n 2 -g 1 -r <i>  (i in [0 - (n - 1)])
if __name__ == "__main__":
    main()

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

"""
This example shows how to execute distributed training based on Horovod APIs.
It can run on several nodes with multiple GPU devices on every node.
Main steps to set up the distributed training:

- Install Horovod referring to the guide: https://github.com/horovod/horovod/blob/master/docs/gpus.rst
  If using MONAI docker, which already has NCCL and MPI, can quickly install Horovod with command:
  `HOROVOD_NCCL_INCLUDE=/usr/include HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_OPERATIONS=NCCL \
  pip install --no-cache-dir horovod`
- Set SSH permissions for root login without password at all nodes except master, referring to:
  http://www.linuxproblem.org/art_9.html
- Run `hvd.init()` to initialize Horovod.
- Pin each GPU to a single process to avoid resource contention, use `hvd.local_rank()` to get GPU index.
  And use `hvd.rank()` to get the overall rank index.
- Wrap Dataset with `DistributedSampler`, and disable the `shuffle` in DataLoader.
  Instead, shuffle data by `train_sampler.set_epoch(epoch)` before every epoch.
- Wrap the optimizer in hvd.DistributedOptimizer. The distributed optimizer delegates gradient
  computation to the original optimizer, averages gradients using allreduce or allgather,
  and then applies those averaged gradients.
- Broadcast the initial variable states from rank 0 to all other processes.

Note:
    Suggest setting exactly the same software environment for every node, especially `mpi`, `nccl`, etc.
    A good practice is to use the same MONAI docker image for all nodes directly, if using docker, need
    to set SSH permissions both at the node and in docker, referring to Horovod guide for more details:
    https://github.com/horovod/horovod/blob/master/docs/docker.rst

    Example script to execute this program, only need to run on the master node:
    `horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python unet_training_horovod.py -d "./testdata"`

    This example was tested with [Ubuntu 16.04/20.04], [NCCL 2.6.3], [horovod 0.19.5].

Referring to: https://github.com/horovod/horovod/blob/master/examples/pytorch_mnist.py

"""

import argparse
import os
import sys
from glob import glob

import horovod.torch as hvd
import nibabel as nib
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import monai
from monai.data import DataLoader, Dataset, create_test_image_3d
from monai.transforms import (
    AsChannelFirstd,
    Compose,
    LoadNiftid,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    ToTensord,
)


def train(args):
    # initialize Horovod library
    hvd.init()
    # Horovod limits CPU threads to be used per worker
    torch.set_num_threads(1)
    # disable logging for processes except 0 on every node
    if hvd.local_rank() != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    elif not os.path.exists(args.dir):
        # create 40 random image, mask paris on master node for training
        print(f"generating synthetic data to {args.dir} (this may take a while)")
        os.makedirs(args.dir)
        # set random seed to generate same random data for every node
        np.random.seed(seed=0)
        for i in range(40):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(args.dir, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(args.dir, f"seg{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(args.dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(args.dir, "seg*.nii.gz")))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    # define transforms for image and segmentation
    train_transforms = Compose(
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

    # create a training data loader
    train_ds = Dataset(data=train_files, transform=train_transforms)
    # create a training data sampler
    train_sampler = DistributedSampler(train_ds, num_replicas=hvd.size(), rank=hvd.rank())
    # when supported, use "forkserver" to spawn dataloader workers instead of "fork" to prevent
    # issues with Infiniband implementations that are not fork-safe
    multiprocessing_context = None
    if hasattr(mp, "_supports_context") and mp._supports_context and "forkserver" in mp.get_all_start_methods():
        multiprocessing_context = "forkserver"
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=train_sampler,
        multiprocessing_context=multiprocessing_context,
    )

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device(f"cuda:{hvd.local_rank()}")
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
    # Horovod broadcasts parameters & optimizer state
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    # Horovod wraps optimizer with DistributedOptimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # start a typical PyTorch training
    epoch_loss_values = list()
    for epoch in range(5):
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
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    print(f"train completed, epoch losses: {epoch_loss_values}")
    if hvd.rank() == 0:
        # all processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes,
        # therefore, saving it in one process is sufficient
        torch.save(model.state_dict(), "final_model.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="./testdata", type=str, help="directory to create random data")
    args = parser.parse_args()

    train(args=args)


# Example script to execute this program only on the master node:
# horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python unet_training_horovod.py -d "./testdata"
if __name__ == "__main__":
    main()

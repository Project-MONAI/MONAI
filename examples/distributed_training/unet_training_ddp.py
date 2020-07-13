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
This example shows how to execute distributed training based on PyTorch native `DistributedDataParallel` module.
It can run on several nodes with multiple GPU devices on every node.
Main steps to set up the distributed training:

- `node` is the node index, `gpus` is GPU count of every node, `gpu` is index of current GPU in 1 node.
- Use `init_process_group` to initialize every process, every GPU runs in a separate process with unique rank.
  Here we use `NVIDIA NCCL` as the backend and get information from environment `env://`.
- Set the IP and PORT for master node.
- Wrap the model with `DistributedDataParallel` after moving to expected device.
- Wrap Dataset with `DistributedSampler`, and disable the `shuffle` and `num_worker=0` in DataLoader.
  Instead, shuffle data by `train_sampler.set_epoch(epoch)` before every epoch.
- Execute the program with `mp.spawn(train, nprocs=args.gpus, args=(args,))` on every node.
  Instead of running the train function once, we will spawn `args.gpus` processes, each of which runs `train(i, args)`,
  where i goes from `0` to `args.gpus - 1`. We run the `main()` function on each node, so that in total there will be
  `args.nodes x args.gpus = args.world_size` processes.
- Every node runs the program with different node index, a typical example for 2 nodes with 1 GPU for every node:
  `python unet_training_ddp.py -n 2 -g 1 -i <i>` (i in [0 - (n - 1)])

Note:
    Every node must has exactly the same software environment, especially `PyTorch`, `nccl`, etc.
    Suggest using the same docker image for all nodes directly.

Referring to: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

"""

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
from torch.nn.parallel import DistributedDataParallel
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
    def _log_print(data, gpu=gpu):
        print(data) if gpu == 0 else None

    # initialize the distributed training process, every GPU runs in a process,
    # so the process rank is (node index x GPU count of 1 node + GPU index)
    rank = args.node * args.gpus + gpu
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
    model = DistributedDataParallel(model, device_ids=[gpu])

    # start a typical PyTorch training
    epoch_loss_values = list()
    for epoch in range(5):
        _log_print("-" * 10)
        _log_print(f"epoch {epoch + 1}/{5}")
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
            _log_print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        _log_print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    _log_print(f"train completed, epoch losses: {epoch_loss_values}")
    if rank == 0:
        torch.save(model.state_dict(), "final_model.pth")
    dist.destroy_process_group()


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    set_determinism(seed=0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nodes", default=1, type=int, help="number of nodes in total")
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("-i", "--node", default=0, type=int, help="node index within all the nodes")
    parser.add_argument("-d", "--dir", default="./testdata", type=str, help="directory to create random data")
    args = parser.parse_args()

    # create 40 random image, mask paris for training
    if not os.path.exists(args.dir):
        print(f"generating synthetic data to {args.dir} (this may take a while)")
        os.makedirs(args.dir)
        for i in range(40):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)

            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(args.dir, f"img{i:d}.nii.gz"))

            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(args.dir, f"seg{i:d}.nii.gz"))

    args.world_size = args.gpus * args.nodes
    os.environ["MASTER_ADDR"] = "10.23.137.29"
    os.environ["MASTER_PORT"] = "8888"
    mp.spawn(train, nprocs=args.gpus, args=(args,))


# usage: "python unet_training_ddp.py -n 2 -g 1 -i <i>"  (i in [0 - (n - 1)])
if __name__ == "__main__":
    main()

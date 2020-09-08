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
This example shows how to execute distributed training based on PyTorch native module and SmartCacheDataset.
It can run on several nodes with multiple GPU devices on every node.
It splits data into partitions, every rank only cache and train with its own partition.

Main steps to set up the distributed training:

- Execute `torch.distributed.launch` to create processes on every node for every GPU.
  It receives parameters as below:
  `--nproc_per_node=NUM_GPUS_PER_NODE`
  `--nnodes=NUM_NODES`
  `--node_rank=INDEX_CURRENT_NODE`
  `--master_addr="192.168.1.1"`
  `--master_port=1234`
  For more details, refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py.
  Alternatively, we can also use `torch.multiprocessing.spawn` to start program, but it that case, need to handle
  all the above parameters and compute `rank` manually, then set to `init_process_group`, etc.
  `torch.distributed.launch` is even more efficient than `torch.multiprocessing.spawn` during training.
- Use `init_process_group` to initialize every process, every GPU runs in a separate process with unique rank.
  Here we use `NVIDIA NCCL` as the backend and must set `init_method="env://"` if use `torch.distributed.launch`.
- Wrap the model with `DistributedDataParallel` after moving to expected device.
- Execute `partition_dataset` to load data only for current rank, no need `DistributedSampler` anymore.
- `SmartCacheDataset` computes and caches the data for the first epoch.
- Call `start()` function of `SmartCacheDataset` to start the replacement thread.
- Call `update_cache()` function of `SmartCacheDataset` before every epoch to replace part of cache content.
- Call `shutdown()` function of `SmartCacheDataset` to stop replacement thread when training ends.

Note:
    `torch.distributed.launch` will launch `nnodes * nproc_per_node = world_size` processes in total.
    Suggest setting exactly the same software environment for every node, especially `PyTorch`, `nccl`, etc.
    A good practice is to use the same MONAI docker image for all nodes directly.
    Example script to execute this program on every node:
    python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
           --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
           --master_addr="192.168.1.1" --master_port=1234
           unet_training_smartcache.py -d DIR_OF_TESTDATA

    This example was tested with [Ubuntu 16.04/20.04], [NCCL 2.6.3].

Referring to: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

"""

import argparse
import math
import os
import sys
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import monai
from monai.data import DataLoader, SmartCacheDataset, create_test_image_3d
from monai.transforms import (
    AsChannelFirstd,
    Compose,
    LoadNiftid,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    ToTensord,
)


def partition_dataset(data, num_replicas=None, rank=None, shuffle=False, seed=0, drop_last=False):
    """
    Partition the dataset for distributed training, every rank process only train with its own data partition.
    It can be useful for `CacheDataset` or `SmartCacheDataset`, because every rank process can only compute and
    cache its own data.
    Note that every rank process will shuffle data only in its own partition if set `shuffle=True` to DataLoader.

    The alternative solution is to use `DistributedSampler`, which supports global shuffle before every epoch.
    But if using `CacheDataset` or `SmartCacheDataset`, every rank process will cache duplicated data content and
    raise system memory usage.

    Args:
        data: data list to partition, assumed to be of constant size.
        num_replicas: number of processes participating in the distributed training.
            if None, retrieve the `world_size` from current distributed group.
        rank: rank of the current process within `num_replicas`.
            if None, retrieve the rank index from current distributed group.
        shuffle: if true, will shuffle the indices of data list before partition.
        seed: random seed to shuffle the indices if `shuffle=True`, default is `0`.
            this number should be identical across all processes in the distributed group.
        drop_last: if `True`, will drop the tail of the data to make it evenly divisible across the number of replicas.
            if `False`, add extra indices to make the data evenly divisible across the replicas. default is `False`.

    """
    if num_replicas is None or rank is None:
        if not dist.is_available():
            raise RuntimeError("require distributed package to be available.")
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

    if drop_last and len(data) % num_replicas != 0:
        # split to nearest available length that is evenly divisible
        num_samples = math.ceil((len(data) - num_replicas) / num_replicas)
    else:
        num_samples = math.ceil(len(data) / num_replicas)
    total_size = num_samples * num_replicas

    indices = np.array(list(range(len(data))))
    if shuffle:
        # deterministically shuffle based on fixed seed for every process
        np.random.seed(seed)
        np.random.shuffle(indices)

    if not drop_last and total_size - len(indices) > 0:
        # add extra samples to make it evenly divisible
        indices += indices[: (total_size - len(indices))]
    else:
        # remove tail of data to make it evenly divisible
        indices = indices[:total_size]

    indices = indices[rank:total_size:num_replicas]
    return [data[i] for i in indices]


def train(args):
    # disable logging for processes except 0 on every node
    if args.local_rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    elif not os.path.exists(args.dir):
        # create 40 random image, mask paris for training
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

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")

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

    # partition dataset based on current rank number, every rank trains with its own data
    data_part = partition_dataset(train_files, shuffle=True)
    train_ds = SmartCacheDataset(
        data=data_part,
        transform=train_transforms,
        replace_rate=0.2,
        cache_num=15,  # we suppose to use 2 ranks in this example, every rank has 20 training images
        num_init_workers=2,
        num_replace_workers=2,
    )
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device(f"cuda:{args.local_rank}")
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
    model = DistributedDataParallel(model, device_ids=[args.local_rank])

    # start a typical PyTorch training
    epoch_loss_values = list()
    # start the replacement thread of SmartCache
    train_ds.start()

    for epoch in range(5):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = math.ceil(len(train_ds) / train_loader.batch_size)
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        # replace 20% of cache content for next epoch
        train_ds.update_cache()
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    # stop replacement thread of SmartCache
    train_ds.shutdown()
    print(f"train completed, epoch losses: {epoch_loss_values}")
    if dist.get_rank() == 0:
        # all processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes,
        # therefore, saving it in one process is sufficient
        torch.save(model.state_dict(), "final_model.pth")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="./testdata", type=str, help="directory to create random data")
    # must parse the command-line argument: ``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by DDP
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    train(args=args)


# usage example(refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py):

# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
#        --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
#        --master_addr="192.168.1.1" --master_port=1234
#        unet_training_smartcache.py -d DIR_OF_TESTDATA

if __name__ == "__main__":
    main()

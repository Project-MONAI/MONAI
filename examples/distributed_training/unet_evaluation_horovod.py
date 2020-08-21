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
This example shows how to execute distributed evaluation based on Horovod APIs.
It can run on several nodes with multiple GPU devices on every node.
Main steps to set up the distributed evaluation:

- Install Horovod referring to the guide: https://github.com/horovod/horovod/blob/master/docs/gpus.rst
  If using MONAI docker, which already has NCCL and MPI, can quickly install Horovod with command:
  `HOROVOD_NCCL_INCLUDE=/usr/include HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_OPERATIONS=NCCL \
  pip install --no-cache-dir horovod`
- Set SSH permissions for root login without password at all nodes except master, referring to:
  http://www.linuxproblem.org/art_9.html
- Run `hvd.init()` to initialize Horovod.
- Pin each GPU to a single process to avoid resource contention, use `hvd.local_rank()` to get GPU index.
  And use `hvd.rank()` to get the overall rank index.
- Wrap Dataset with `DistributedSampler`, disable `shuffle` for sampler and DataLoader.
- Broadcast the model parameters from rank 0 to all other processes.

Note:
    Suggest setting exactly the same software environment for every node, especially `mpi`, `nccl`, etc.
    A good practice is to use the same MONAI docker image for all nodes directly, if using docker, need
    to set SSH permissions both at the node and in docker, referring to Horovod guide for more details:
    https://github.com/horovod/horovod/blob/master/docs/docker.rst

    Example script to execute this program, only need to run on the master node:
    `horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python unet_evaluation_horovod.py -d "./testdata"`

    This example was tested with [Ubuntu 16.04/20.04], [NCCL 2.6.3], [horovod 0.19.5].

Referring to: https://github.com/horovod/horovod/blob/master/examples/pytorch_mnist.py

"""

import argparse
import os
from glob import glob

import horovod.torch as hvd
import nibabel as nib
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import monai
from monai.data import DataLoader, Dataset, create_test_image_3d
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsChannelFirstd, Compose, LoadNiftid, ScaleIntensityd, ToTensord


def evaluate(args):
    # initialize Horovod library
    hvd.init()
    # Horovod limits CPU threads to be used per worker
    torch.set_num_threads(1)

    if hvd.local_rank() == 0 and not os.path.exists(args.dir):
        # create 16 random image, mask paris for evaluation
        print(f"generating synthetic data to {args.dir} (this may take a while)")
        os.makedirs(args.dir)
        # set random seed to generate same random data for every node
        np.random.seed(seed=0)
        for i in range(16):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(args.dir, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(args.dir, f"seg{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(args.dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(args.dir, "seg*.nii.gz")))
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    # define transforms for image and segmentation
    val_transforms = Compose(
        [
            LoadNiftid(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys="img"),
            ToTensord(keys=["img", "seg"]),
        ]
    )

    # create a evaluation data loader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    # create a evaluation data sampler
    val_sampler = DistributedSampler(val_ds, shuffle=False, num_replicas=hvd.size(), rank=hvd.rank())
    # when supported, use "forkserver" to spawn dataloader workers instead of "fork" to prevent
    # issues with Infiniband implementations that are not fork-safe
    multiprocessing_context = None
    if hasattr(mp, "_supports_context") and mp._supports_context and "forkserver" in mp.get_all_start_methods():
        multiprocessing_context = "forkserver"
    # sliding window inference need to input 1 image in every iteration
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=val_sampler,
        multiprocessing_context=multiprocessing_context,
    )
    dice_metric = DiceMetric(include_background=True, to_onehot_y=False, sigmoid=True, reduction="mean")

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
    if hvd.rank() == 0:
        # load model parameters for evaluation
        model.load_state_dict(torch.load("final_model.pth"))
    # Horovod broadcasts parameters
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model.eval()
    with torch.no_grad():
        # define PyTorch Tensor to record metrics result at each GPU
        # the first value is `sum` of all dice metric, the second value is `count` of not_nan items
        metric = torch.zeros(2, dtype=torch.float, device=device)
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            value = dice_metric(y_pred=val_outputs, y=val_labels).squeeze()
            metric[0] += value * dice_metric.not_nans
            metric[1] += dice_metric.not_nans
        # synchronizes all processes and reduce results
        print(f"metric in rank {hvd.rank()}: sum={metric[0].item()}, count={metric[1].item()}")
        avg_metric = hvd.allreduce(metric, name="mean_dice")
        if hvd.rank() == 0:
            print(f"average metric: sum={avg_metric[0].item()}, count={avg_metric[1].item()}")
            print("evaluation metric:", (avg_metric[0] / avg_metric[1]).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="./testdata", type=str, help="directory to create random data")
    args = parser.parse_args()

    evaluate(args=args)


# Example script to execute this program only on the master node:
# horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python unet_evaluation_horovod.py -d "./testdata"
if __name__ == "__main__":
    main()

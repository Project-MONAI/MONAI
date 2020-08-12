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
This example shows how to execute distributed evaluation based on PyTorch native `DistributedDataParallel` module.
It can run on several nodes with multiple GPU devices on every node.
Main steps to set up the distributed evaluation:

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
  `torch.distributed.launch` is even more efficient than `torch.multiprocessing.spawn`.
- Use `init_process_group` to initialize every process, every GPU runs in a separate process with unique rank.
  Here we use `NVIDIA NCCL` as the backend and must set `init_method="env://"` if use `torch.distributed.launch`.
- Wrap the model with `DistributedDataParallel` after moving to expected device.
- Put model file on every node, then load and map to expected GPU device in every process.
- Wrap Dataset with `DistributedSampler`, disable the `shuffle` in sampler and DataLoader.
- Compute `Dice Metric` on every process, reduce the results after synchronization.

Note:
    `torch.distributed.launch` will launch `nnodes * nproc_per_node = world_size` processes in total.
    Suggest setting exactly the same software environment for every node, especially `PyTorch`, `nccl`, etc.
    A good practice is to use the same MONAI docker image for all nodes directly.
    Example script to execute this program on every node:
    python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
           --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
           --master_addr="192.168.1.1" --master_port=1234
           unet_evaluation_ddp.py -d DIR_OF_TESTDATA

    This example was tested with [Ubuntu 16.04/20.04], [NCCL 2.6.3].

Referring to: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

"""

import argparse
import os
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import monai
from monai.data import DataLoader, Dataset, create_test_image_3d
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsChannelFirstd, Compose, LoadNiftid, ScaleIntensityd, ToTensord


def evaluate(args):
    if args.local_rank == 0 and not os.path.exists(args.dir):
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

    # initialize the distributed evaluation process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")

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
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    # sliding window inference need to input 1 image in every iteration
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, sampler=val_sampler)
    dice_metric = DiceMetric(include_background=True, to_onehot_y=False, sigmoid=True, reduction="mean")

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
    # wrap the model with DistributedDataParallel module
    model = DistributedDataParallel(model, device_ids=[args.local_rank])
    # config mapping to expected GPU device
    map_location = {"cuda:0": f"cuda:{args.local_rank}"}
    # load model parameters to GPU device
    model.load_state_dict(torch.load("final_model.pth", map_location=map_location))

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
        dist.barrier()
        dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
        metric = metric.tolist()
        if dist.get_rank() == 0:
            print("evaluation metric:", metric[0] / metric[1])
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="./testdata", type=str, help="directory to create random data")
    # must parse the command-line argument: ``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by DDP
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    evaluate(args=args)


# usage example(refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py):

# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
#        --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
#        --master_addr="192.168.1.1" --master_port=1234
#        unet_evaluation_ddp.py -d DIR_OF_TESTDATA

if __name__ == "__main__":
    main()

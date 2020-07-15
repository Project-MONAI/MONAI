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

- `node` is the node index, `gpus` is GPU count of every node, `gpu` is index of current GPU in 1 node.
- Use `init_process_group` to initialize every process, every GPU runs in a separate process with unique rank.
  Here we use `NVIDIA NCCL` as the backend and get information from environment `env://`.
- Set the IP and PORT for master node.
- Wrap the model with `DistributedDataParallel` after moving to expected device.
- Load model parameters from local file and map to expected GPU device in every process.
- Wrap Dataset with `DistributedSampler`, set `num_worker=0` in DataLoader.
- Compute `Dice Metric` on every process, reduce the results after synchronization.
- Execute the program with `mp.spawn(evaluate, nprocs=args.gpus, args=(args,))` on every node.
  Instead of running the `evaluate` function once, we will spawn `args.gpus` processes,
  each of which runs `evaluate(i, args)`, where i goes from `0` to `args.gpus - 1`. We run the `main()`
  function on each node, so that in total there will be `args.nodes x args.gpus = args.world_size` processes.
- Every node runs the program with different node index, a typical example for 2 nodes with 1 GPU for every node:
  `python unet_evaluation_ddp.py -mi 10.23.137.29 -mp 8888 -n 2 -g 1 -i <i>` (i in [0 ~ (n - 1)])

Note:
    Suggest setting exactly the same software environment for every node, especially `PyTorch`, `nccl`, etc.
    A good practice is to use the same MONAI docker image for all nodes directly.

Referring to: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

"""

import os
from glob import glob
import nibabel as nib
import numpy as np
import torch
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import monai
from monai.transforms import (
    Compose,
    LoadNiftid,
    AsChannelFirstd,
    ScaleIntensityd,
    ToTensord,
)
from monai.data import create_test_image_3d, Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric


def evaluate(gpu, args):
    # initialize the distributed evaluation process, every GPU runs in a process,
    # so the process rank is (node index x GPU count of 1 node + GPU index)
    rank = args.node * args.gpus + gpu
    dist.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=rank)

    images = sorted(glob(os.path.join(args.dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(args.dir, "seg*.nii.gz")))
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

    # create a evaluation data loader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    # create a evaluation data sampler
    val_sampler = DistributedSampler(val_ds, num_replicas=args.world_size, rank=rank)
    # sliding window inference need to input 1 image in every iteration
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(), sampler=val_sampler,
    )
    dice_metric = DiceMetric(include_background=True, to_onehot_y=False, sigmoid=True, reduction="mean")

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
    # wrap the model with DistributedDataParallel module
    model = DistributedDataParallel(model, device_ids=[gpu])
    # config mapping to expected GPU device
    map_location = {"cuda:0": f"cuda:{gpu}"}
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
        if rank == 0:
            print("evaluation metric:", metric[0] / metric[1])
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mi", "--master_ip", default="localhost", type=str, help="IP address of the master node")
    parser.add_argument("-mp", "--master_port", default="8888", type=str, help="PORT of the master node")
    parser.add_argument("-n", "--nodes", default=1, type=int, help="number of nodes in total")
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("-i", "--node", default=0, type=int, help="node index within all the nodes")
    parser.add_argument("-d", "--dir", default="./testdata", type=str, help="directory to create random data")
    args = parser.parse_args()

    # create 16 random image, mask paris for evaluation
    if not os.path.exists(args.dir):
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

    args.world_size = args.gpus * args.nodes
    os.environ["MASTER_ADDR"] = args.master_ip
    os.environ["MASTER_PORT"] = args.master_port
    mp.spawn(evaluate, nprocs=args.gpus, args=(args,))


# usage: "python unet_evaluation_ddp.py -mi 10.23.137.29 -mp 8888 -n 2 -g 1 -i <i>"  (i in [0 - (n - 1)])
if __name__ == "__main__":
    main()

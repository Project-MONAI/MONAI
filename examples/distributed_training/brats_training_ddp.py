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
- Wrap Dataset with `DistributedSampler`, and disable the `shuffle` and `num_worker=0` in DataLoader.
  Instead, shuffle data by `train_sampler.set_epoch(epoch)` before every epoch.

Note:
    `torch.distributed.launch` will launch `nnodes * nproc_per_node = world_size` processes in total.
    Suggest setting exactly the same software environment for every node, especially `PyTorch`, `nccl`, etc.
    A good practice is to use the same MONAI docker image for all nodes directly.
    Example script to execute this program on every node:
    python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
           --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
           --master_addr="192.168.1.1" --master_port=1234
           unet_training_ddp.py -d DIR_OF_TESTDATA

    This example was tested with [Ubuntu 16.04/20.04], [NCCL 2.6.3].

Referring to: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

Some codes are taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py

"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AsChannelFirstd,
    CenterSpatialCropd,
    Compose,
    LoadNiftid,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
)
from monai.utils import set_determinism


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WC (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = list()
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WC
            result.append(np.logical_or(np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


def main_worker(args):
    # disable logging for processes except 0 on every node
    if args.local_rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    if not os.path.exists(args.dir):
        raise FileNotFoundError(f'Missing directory {args.dir}')

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")

    train_transforms = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadNiftid(keys=["image", "label"]),
            AsChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64], random_size=False),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # create a training data loader
    train_ds = DecathlonDataset(
        root_dir=args.dir,
        task="Task01_BrainTumour",
        transform=train_transforms,
        section="training",
        download=False,
        num_workers=4,
        cache_rate=1.0
    )
    # create a training data sampler
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler,
    )

    if dist.get_rank() == 0:
        val_transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AsChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                ToTensord(keys=["image", "label"]),
            ]
        )
        val_ds = DecathlonDataset(
            root_dir=args.dir,
            task="Task01_BrainTumour",
            transform=val_transforms,
            section="validation",
            download=False,
            num_workers=4,
            cache_rate=1.0
        )
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=args.workers)

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device(f"cuda:{args.local_rank}")
    model = UNet(
        dimensions=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)
    # wrap the model with DistributedDataParallel module
    model = DistributedDataParallel(model, device_ids=[args.local_rank])

    # start a typical PyTorch training
    total_epoch = args.epochs
    best_metric = -1000000
    best_metric_epoch = -1
    epoch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        total_epoch,
        [epoch_time],
        prefix="Epoch: ")
    end = time.time()
    for epoch in range(total_epoch):
        train_sampler.set_epoch(epoch)

        train(train_loader, model, loss_function, optimizer, epoch, args, device)
        epoch_time.update(time.time() - end)

        if epoch % args.print_freq == 0:
            progress.display(epoch)

        if dist.get_rank() == 0 and (epoch + 1) % args.val_interval == 0:
            metric, metric_tc, metric_wt, metric_et = evaluate(model, val_loader, device)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                # torch.save(model.state_dict(), os.path.join(args.dir, "best_metric_model.pth"))
                # print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )
        end = time.time()

    if dist.get_rank() == 0:
        print(f"train completed, best_metric: {best_metric:.4f}  at epoch: {best_metric_epoch}")
        # all processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes,
        # therefore, saving it in one process is sufficient
        torch.save(model.state_dict(), "final_model.pth")
    dist.destroy_process_group()


def train(train_loader, model, criterion, optimizer, epoch, args, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch_data in enumerate(train_loader):
        image = batch_data['image'].to(device, non_blocking=True)
        target = batch_data['label'].to(device, non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)

        # record loss
        losses.update(loss.item(), image.size(0))

        # compute gradient and do GD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)


def evaluate(model, data_loader, device):
    metric_values = list()
    metric_values_tc = list()
    metric_values_wt = list()
    metric_values_et = list()
    metric_sum = metric_sum_tc = metric_sum_wt = metric_sum_et = 0.0
    metric_count = metric_count_tc = metric_count_wt = metric_count_et = 0

    model.eval()
    with torch.no_grad():
        dice_metric = DiceMetric(include_background=True, sigmoid=True, reduction="mean")
        for val_data in data_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            val_outputs = model(val_inputs)
            # compute overall mean dice
            value = dice_metric(y_pred=val_outputs, y=val_labels)
            not_nans = dice_metric.not_nans.item()
            metric_count += not_nans
            metric_sum += value.item() * not_nans
            # compute mean dice for TC
            value_tc = dice_metric(y_pred=val_outputs[:, 0:1], y=val_labels[:, 0:1])
            not_nans = dice_metric.not_nans.item()
            metric_count_tc += not_nans
            metric_sum_tc += value_tc.item() * not_nans
            # compute mean dice for WT
            value_wt = dice_metric(y_pred=val_outputs[:, 1:2], y=val_labels[:, 1:2])
            not_nans = dice_metric.not_nans.item()
            metric_count_wt += not_nans
            metric_sum_wt += value_wt.item() * not_nans
            # compute mean dice for ET
            value_et = dice_metric(y_pred=val_outputs[:, 2:3], y=val_labels[:, 2:3])
            not_nans = dice_metric.not_nans.item()
            metric_count_et += not_nans
            metric_sum_et += value_et.item() * not_nans

        metric = metric_sum / metric_count
        metric_values.append(metric)
        metric_tc = metric_sum_tc / metric_count_tc
        metric_values_tc.append(metric_tc)
        metric_wt = metric_sum_wt / metric_count_wt
        metric_values_wt.append(metric_wt)
        metric_et = metric_sum_et / metric_count_et
        metric_values_et.append(metric_et)
    return metric, metric_tc, metric_wt, metric_et


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="./testdata", type=str, help="directory to create random data")
    # must parse the command-line argument: ``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by DDP
    parser.add_argument("--local_rank", type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_world_size", type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print_freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument("--cache_rate", type=float, default=1.0)
    parser.add_argument("--val_interval", type=int, default=5)
    args = parser.parse_args()

    if args.seed is not None:
        set_determinism(seed=args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args=args)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# usage example(refer to https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py):

# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
#        --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
#        --master_addr="10.110.44.150" --master_port=1234
#        msd_brats_training_ddp.py -d DIR_OF_TESTDATA


if __name__ == "__main__":
    main()

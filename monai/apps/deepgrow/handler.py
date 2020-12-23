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

import logging
import os
import statistics

import nibabel as nib
import numpy as np
import torch
import torch.distributed
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from monai.engines.workflow import Engine, Events
from monai.metrics import compute_meandice
from monai.transforms import rescale_array
from monai.visualize import plot_2d_or_3d_image

from .utils import make_grid_with_titles

# TODO:: Unit Test


class MeanDice:
    def __init__(self):
        self.data = []

    def reset(self):
        self.data = []

    def update(self, y_pred, y, batched=True):
        if not batched:
            y_pred = y_pred[None]
            y = y[None]
        score = compute_meandice(y_pred=y_pred, y=y, include_background=False).mean()
        self.data.append(score.item())

    def mean(self) -> float:
        return statistics.mean(self.data)

    def stdev(self) -> float:
        return statistics.stdev(self.data) if len(self.data) > 1 else 0


class DeepgrowStatsHandler(object):
    def __init__(
        self,
        summary_writer=None,
        interval=1,
        log_dir="./runs",
        tag_name="val_dice",
        compute_metric=True,
        images=True,
        image_interval=1,
        max_channels=1,
        max_frames=64,
        add_scalar=True,
        add_stdev=False,
        merge_scalar=False,
        fold_size=0,
    ):
        self.writer = SummaryWriter(log_dir=log_dir) if summary_writer is None else summary_writer
        self.interval = interval
        self.tag_name = tag_name
        self.compute_metric = compute_metric
        self.images = images
        self.image_interval = image_interval
        self.max_channels = max_channels
        self.max_frames = max_frames
        self.add_scalar = add_scalar
        self.add_stdev = add_stdev
        self.merge_scalar = merge_scalar
        self.fold_size = fold_size

        if torch.distributed.is_initialized():
            self.tag_name = "{}-r{}".format(self.tag_name, torch.distributed.get_rank())

        self.plot_data = {}
        self.metric_data = {}

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self, "iteration")
        engine.add_event_handler(Events.EPOCH_COMPLETED(every=1), self, "epoch")

    def write_images(self, epoch):
        if not self.plot_data or not len(self.plot_data):
            return

        all_imgs = []
        titles = []
        for region in sorted(self.plot_data.keys()):
            all_imgs.extend(self.plot_data[region])
            metric = self.metric_data.get(region)
            dice = "{:.4f}".format(metric.mean()) if self.compute_metric and metric else ""
            stdev = "{:.4f}".format(metric.stdev()) if self.compute_metric and metric else ""
            titles.extend(
                [
                    "x({})".format(region),
                    "y({})".format(region),
                    "dice: {} +/- {}".format(dice, stdev) if self.compute_metric else "yh({})".format(region),
                ]
            )

        if len(all_imgs[0].shape) == 3:
            colors = [(0, 0, 255), (0, 0, 255), (255, 0, 0)]
            img_tensor = make_grid_with_titles(
                tensor=torch.from_numpy(np.array(all_imgs)),
                titles=titles,
                colors=colors,
                nrow=3,
                normalize=True,
                pad_value=2,
            )
            self.writer.add_image(tag=f"Deepgrow Regions ({self.tag_name})", img_tensor=img_tensor, global_step=epoch)

        if len(all_imgs[0].shape) == 4:
            for region in sorted(self.plot_data.keys()):
                tags = [f"region_{region}_image", f"region_{region}_label", f"region_{region}_output"]
                for i in range(3):
                    img = self.plot_data[region][i]
                    plot_2d_or_3d_image(
                        img[np.newaxis], epoch, self.writer, 0, self.max_channels, self.max_frames, tags[i]
                    )

        logging.info(
            "Saved {} Regions {} into Tensorboard at epoch: {}".format(
                len(self.plot_data), sorted([*self.plot_data]), epoch
            )
        )
        self.writer.flush()

    def write_region_metrics(self, epoch):
        metric_sum = 0
        means = {}
        stdevs = {}
        for region in self.metric_data:
            metric = self.metric_data[region].mean()
            stdev = self.metric_data[region].stdev()
            if self.merge_scalar:
                means["{:0>2d}".format(region)] = metric
                stdevs["{:0>2d}".format(region)] = stdev
            else:
                if self.add_stdev:
                    self.writer.add_scalar("{}_{:0>2d}_mean".format(self.tag_name, region), metric, epoch)
                    self.writer.add_scalar("{}_{:0>2d}_mean+".format(self.tag_name, region), metric + stdev, epoch)
                    self.writer.add_scalar("{}_{:0>2d}_mean-".format(self.tag_name, region), metric - stdev, epoch)
                else:
                    self.writer.add_scalar("{}_{:0>2d}".format(self.tag_name, region), metric, epoch)
            metric_sum += metric
        if self.merge_scalar:
            self.writer.add_scalars("{}_region".format(self.tag_name), means, epoch)

        if len(self.metric_data) > 1:
            metric_avg = metric_sum / len(self.metric_data)
            self.writer.add_scalar("{}_regions_avg".format(self.tag_name), metric_avg, epoch)
        self.writer.flush()

    def __call__(self, engine: Engine, action) -> None:
        total_steps = engine.state.iteration
        if total_steps < engine.state.epoch_length:
            total_steps = engine.state.epoch_length * (engine.state.epoch - 1) + total_steps

        if action == "epoch" and not self.fold_size:
            epoch = engine.state.epoch
        elif self.fold_size and total_steps % self.fold_size == 0:
            epoch = int(total_steps / self.fold_size)
        else:
            epoch = None

        if epoch:
            if self.images and epoch % self.image_interval == 0:
                self.write_images(epoch)
            if self.add_scalar:
                self.write_region_metrics(epoch)

        if action == "epoch" or epoch:
            self.plot_data = {}
            self.metric_data = {}
            return

        device = engine.state.device
        batch_data = engine.state.batch
        output_data = engine.state.output

        for bidx in range(len(batch_data.get("region", []))):
            region = batch_data.get("region")[bidx]
            region = region.item() if torch.is_tensor(region) else region

            if self.images and self.plot_data.get(region) is None:
                self.plot_data[region] = [
                    rescale_array(batch_data["image"][bidx][0].detach().cpu().numpy()[np.newaxis], 0, 1),
                    rescale_array(batch_data["label"][bidx].detach().cpu().numpy(), 0, 1),
                    rescale_array(output_data["pred"][bidx].detach().cpu().numpy(), 0, 1),
                ]

            if self.compute_metric:
                if self.metric_data.get(region) is None:
                    self.metric_data[region] = MeanDice()
                self.metric_data[region].update(
                    y_pred=output_data["pred"][bidx].to(device), y=batch_data["label"][bidx].to(device), batched=False
                )


class SegmentationSaver:
    def __init__(
        self,
        output_dir: str = "./runs",
        save_np=False,
        images=True,
    ):
        self.output_dir = output_dir
        self.save_np = save_np
        self.images = images
        os.makedirs(self.output_dir, exist_ok=True)

    def attach(self, engine: Engine) -> None:
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine: Engine):
        batch_data = engine.state.batch
        output_data = engine.state.output
        device = engine.state.device
        tag = ""
        if torch.distributed.is_initialized():
            tag = "r{}-".format(torch.distributed.get_rank())

        for bidx in range(len(batch_data.get("image"))):
            step = engine.state.iteration
            region = batch_data.get("region")[bidx]
            region = region.item() if torch.is_tensor(region) else region

            image = batch_data["image"][bidx][0].detach().cpu().numpy()[np.newaxis]
            label = batch_data["label"][bidx].detach().cpu().numpy()
            pred = output_data["pred"][bidx].detach().cpu().numpy()
            dice = compute_meandice(
                y_pred=output_data["pred"][bidx][None].to(device),
                y=batch_data["label"][bidx][None].to(device),
                include_background=False,
            ).mean()

            if self.save_np:
                np.savez(
                    os.path.join(
                        self.output_dir,
                        "{}img_label_pred_{}_{:0>4d}_{:0>2d}_{:.4f}".format(tag, region, step, bidx, dice),
                    ),
                    image,
                    label,
                    pred,
                )

            if self.images and len(image.shape) == 3:
                img = make_grid(torch.from_numpy(rescale_array(image, 0, 1)[0]))
                lab = make_grid(torch.from_numpy(rescale_array(label, 0, 1)[0]))

                pos = rescale_array(output_data["image"][bidx][1].detach().cpu().numpy()[np.newaxis], 0, 1)[0]
                neg = rescale_array(output_data["image"][bidx][2].detach().cpu().numpy()[np.newaxis], 0, 1)[0]
                pre = make_grid(torch.from_numpy(np.array([rescale_array(pred, 0, 1)[0], pos, neg])))

                torchvision.utils.save_image(
                    tensor=[img, lab, pre],
                    nrow=3,
                    pad_value=2,
                    fp=os.path.join(
                        self.output_dir,
                        "{}img_label_pred_{}_{:0>4d}_{:0>2d}_{:.4f}.png".format(tag, region, step, bidx, dice),
                    ),
                )

            if self.images and len(image.shape) == 4:
                samples = {"image": image[0], "label": label[0], "pred": pred[0]}
                for sample in samples:
                    img = nib.Nifti1Image(samples[sample], np.eye(4))
                    nib.save(
                        img,
                        os.path.join(
                            self.output_dir, "{}{}_{:0>4d}_{:0>2d}_{:.4f}.nii.gz".format(tag, sample, step, bidx, dice)
                        ),
                    )

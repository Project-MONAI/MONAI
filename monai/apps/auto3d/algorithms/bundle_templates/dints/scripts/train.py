# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import monai
import numpy as np
import os
import random
import sys
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

from datetime import datetime
from monai import transforms
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import ThreadDataLoader, partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Sequence, Union


def run(
    config_file: Optional[Union[str, Sequence[str]]] = None,
    **override,
):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    _args = _update_args(config_file=config_file, **override)
    config_file_ = _pop_args(_args, "config_file")[0]

    parser = ConfigParser()
    parser.read_config(config_file_)
    parser.update(pairs=_args)

    amp = parser.get_parsed_content("amp")
    ckpt_path = parser.get_parsed_content("ckpt_path")
    data_file_base_dir = parser.get_parsed_content("data_file_base_dir")
    data_list_file_path = parser.get_parsed_content("data_list_file_path")
    determ = parser.get_parsed_content("determ")
    finetune = parser.get_parsed_content("finetune")
    fold = parser.get_parsed_content("fold")
    learning_rate = parser.get_parsed_content("learning_rate")
    num_images_per_batch = parser.get_parsed_content("num_images_per_batch")
    num_iterations = parser.get_parsed_content("num_iterations")
    num_iterations_per_validation = parser.get_parsed_content("num_iterations_per_validation")
    num_sw_batch_size = parser.get_parsed_content("num_sw_batch_size")
    output_classes = parser.get_parsed_content("output_classes")
    overlap_ratio = parser.get_parsed_content("overlap_ratio")
    patch_size_valid = parser.get_parsed_content("patch_size_valid")
    softmax = parser.get_parsed_content("softmax")

    train_transforms = parser.get_parsed_content("transforms_train")
    val_transforms = parser.get_parsed_content("transforms_validate")

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)

    if determ:
        set_determinism(seed=0)

    print("[info] number of GPUs:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
    else:
        world_size = 1
    print("[info] world_size:", world_size)

    with open(data_list_file_path, "r") as f:
        json_data = json.load(f)

    list_train = []
    list_valid = []
    for item in json_data["training"]:
        if item["fold"] == fold:
            item.pop("fold", None)
            list_valid.append(item)
        else:
            item.pop("fold", None)
            list_train.append(item)

    files = []
    for _i in range(len(list_train)):
        str_img = os.path.join(data_file_base_dir, list_train[_i]["image"])
        str_seg = os.path.join(data_file_base_dir, list_train[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})

    train_files = files
    random.shuffle(train_files)

    if torch.cuda.device_count() > 1:
        train_files = partition_dataset(
            data=train_files,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=True,
        )[dist.get_rank()]
    print("train_files:", len(train_files))

    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(data_file_base_dir, list_valid[_i]["image"])
        str_seg = os.path.join(data_file_base_dir, list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})

    val_files = files

    if torch.cuda.device_count() > 1:
        if len(val_files) < world_size:
            val_files = val_files * math.ceil(float(world_size) / float(len(val_files)))

        val_files = partition_dataset(
            data=val_files,
            shuffle=False,
            num_partitions=world_size,
            even_divisible=False,
        )[dist.get_rank()]
    print("val_files:", len(val_files))

    if torch.cuda.device_count() >= 4:
        train_ds = monai.data.CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=8
        )
        val_ds = monai.data.CacheDataset(
            data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2
        )
    else:
        train_ds = monai.data.CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=float(torch.cuda.device_count()) / 4.0,
            num_workers=8,
        )
        val_ds = monai.data.CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=float(torch.cuda.device_count()) / 4.0,
            num_workers=2,
        )

    train_loader = ThreadDataLoader(
        train_ds, num_workers=12, batch_size=num_images_per_batch, shuffle=True
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1, shuffle=False)

    device = torch.device(f"cuda:{dist.get_rank()}") if torch.cuda.device_count() > 1 else torch.device("cuda:0")
    torch.cuda.set_device(device)

    model = parser.get_parsed_content("network")
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if softmax:
        post_pred = transforms.Compose(
            [
                transforms.EnsureType(),
                transforms.AsDiscrete(argmax=True, to_onehot=output_classes),
            ]
        )
        post_label = transforms.Compose(
            [
                transforms.EnsureType(),
                transforms.AsDiscrete(to_onehot=output_classes),
            ]
        )
    else:
        post_pred = transforms.Compose(
            [
                transforms.EnsureType(),
                transforms.Activations(sigmoid=True),
                transforms.AsDiscrete(threshold=0.5),
            ]
        )

    loss_function = parser.get_parsed_content("loss")

    optimizer_part = parser.get_parsed_content("optimizer", instantiate=False)
    optimizer = optimizer_part.instantiate(params=model.parameters())

    num_epochs_per_validation = num_iterations_per_validation // len(train_loader)
    num_epochs_per_validation = max(num_epochs_per_validation, 1)
    num_epochs = num_epochs_per_validation * (
        num_iterations // num_iterations_per_validation
    )

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        print("num_epochs", num_epochs)
        print("num_epochs_per_validation", num_epochs_per_validation)

    lr_scheduler_part = parser.get_parsed_content("lr_scheduler", instantiate=True)
    lr_scheduler = lr_scheduler_part.instantiate(optimizer=optimizer)

    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
        )

    if finetune["activate"] and os.path.isfile(finetune["pretrained_ckpt_name"]):
        print("[info] fine-tuning pre-trained checkpoint {0:s}".format(finetune["pretrained_ckpt_name"]))
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(finetune["pretrained_ckpt_name"], map_location=device))
        else:
            model.load_state_dict(torch.load(finetune["pretrained_ckpt_name"], map_location=device))
    else:
        print("[info] training from scratch")

    if amp:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            print("[info] amp enabled")

    val_interval = num_epochs_per_validation
    best_metric = -1
    best_metric_epoch = -1
    idx_iter = 0
    metric_dim = output_classes - 1 if softmax else output_classes

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(ckpt_path, "Events"))

        with open(os.path.join(ckpt_path, "accuracy_history.csv"), "a") as f:
            f.write("epoch\tmetric\tloss\tlr\ttime\titer\n")

    start_time = time.time()
    for epoch in range(num_epochs):
        lr = lr_scheduler.get_last_lr()[0]
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{num_epochs}")
            print("learning rate is set to {}".format(lr))

        model.train()
        epoch_loss = 0
        loss_torch = torch.zeros(2, dtype=torch.float, device=device)
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(
                device
            )

            for param in model.parameters():
                param.grad = None

            if amp:
                with autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs.float(), labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_function(outputs.float(), labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            epoch_loss += loss.item()
            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0
            epoch_len = len(train_loader)
            idx_iter += 1

            if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                print(
                    "[{0}] ".format(str(datetime.now())[:19])
                    + f"{step}/{epoch_len}, train_loss: {loss.item():.4f}"
                )
                writer.add_scalar("Loss/train", loss.item(), epoch_len * epoch + step)

            lr_scheduler.step()

        if torch.cuda.device_count() > 1:
            dist.barrier()
            dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            print(
                f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}, "
                f"best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
            )

        if (epoch + 1) % val_interval == 0 or (epoch + 1) == num_epochs:
            torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                metric = torch.zeros(metric_dim * 2, dtype=torch.float, device=device)
                metric_sum = 0.0
                metric_count = 0
                metric_mat = []
                val_images = None
                val_labels = None
                val_outputs = None

                _index = 0
                for val_data in val_loader:
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    with torch.cuda.amp.autocast(enabled=amp):
                        val_outputs = sliding_window_inference(
                            val_images,
                            patch_size_valid,
                            num_sw_batch_size,
                            model,
                            mode="gaussian",
                            overlap=overlap_ratio,
                        )

                    val_outputs = post_pred(val_outputs[0, ...])
                    val_outputs = val_outputs[None, ...]

                    if softmax:
                        val_labels = post_label(val_labels[0, ...])
                        val_labels = val_labels[None, ...]

                    value = compute_meandice(
                        y_pred=val_outputs, y=val_labels, include_background=not softmax
                    )

                    print(_index + 1, "/", len(val_loader), value)

                    metric_count += len(value)
                    metric_sum += value.sum().item()
                    metric_vals = value.cpu().numpy()
                    if len(metric_mat) == 0:
                        metric_mat = metric_vals
                    else:
                        metric_mat = np.concatenate((metric_mat, metric_vals), axis=0)

                    for _c in range(metric_dim):
                        val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                        val1 = 1.0 - torch.isnan(value[0, 0]).float()
                        metric[2 * _c] += val0 * val1
                        metric[2 * _c + 1] += val1

                    _index += 1

                if torch.cuda.device_count() > 1:
                    dist.barrier()
                    dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)

                metric = metric.tolist()
                if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
                    for _c in range(metric_dim):
                        print(
                            "evaluation metric - class {0:d}:".format(_c + 1),
                            metric[2 * _c] / metric[2 * _c + 1],
                        )
                    avg_metric = 0
                    for _c in range(metric_dim):
                        avg_metric += metric[2 * _c] / metric[2 * _c + 1]
                    avg_metric = avg_metric / float(metric_dim)
                    print("avg_metric", avg_metric)

                    writer.add_scalar("Accuracy/validation", avg_metric, epoch)

                    if avg_metric > best_metric:
                        best_metric = avg_metric
                        best_metric_epoch = epoch + 1
                        if torch.cuda.device_count() > 1:
                            torch.save(
                                model.module.state_dict(),
                                os.path.join(ckpt_path, "best_metric_model.pt"),
                            )
                        else:
                            torch.save(
                                model.state_dict(),
                                os.path.join(ckpt_path, "best_metric_model.pt"),
                            )
                        print("saved new best metric model")

                        dict_file = {}
                        dict_file["best_avg_dice_score"] = float(best_metric)
                        dict_file["best_avg_dice_score_epoch"] = int(best_metric_epoch)
                        dict_file["best_avg_dice_score_iteration"] = int(idx_iter)
                        with open(
                            os.path.join(ckpt_path, "progress.yaml"), "a"
                        ) as out_file:
                            documents = yaml.dump(dict_file, stream=out_file)

                    print(
                        "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, avg_metric, best_metric, best_metric_epoch
                        )
                    )

                    current_time = time.time()
                    elapsed_time = (current_time - start_time) / 60.0
                    with open(
                        os.path.join(ckpt_path, "accuracy_history.csv"), "a"
                    ) as f:
                        f.write(
                            "{0:d}\t{1:.5f}\t{2:.5f}\t{3:.5f}\t{4:.1f}\t{5:d}\n".format(
                                epoch + 1,
                                avg_metric,
                                loss_torch_epoch,
                                lr,
                                elapsed_time,
                                idx_iter,
                            )
                        )

                if torch.cuda.device_count() > 1:
                    dist.barrier()

            torch.cuda.empty_cache()

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        writer.flush()
        writer.close()

    if torch.cuda.device_count() > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()

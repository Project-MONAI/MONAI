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

import time
from argparse import ArgumentParser
import os

import numpy as np
import torch
from monai.transforms import AddChannelDict, Compose, RandCropByPosNegLabeld, Rand3DElasticd, SpatialPadd
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import compute_meandice
from monai.data import Dataset, list_data_collate
from monai.utils import first
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_size

from unet_pipe import UNetPipe, flatten_sequential
from data_utils import get_filenames, load_data_and_mask

N_CLASSES = 10
TRAIN_PATH = "./data/HaN/train/"  # training data folder
VAL_PATH = "./data/HaN/test/"  # validation data folder

torch.backends.cudnn.enabled = True


class ImageLabelDataset:
    """
    Load image and multi-class labels based on the predefined folder structure.
    """

    def __init__(self, path, n_class=10):
        self.path = path
        self.data = sorted(os.listdir(path))
        self.n_class = n_class

    def __getitem__(self, index):
        data = os.path.join(self.path, self.data[index])
        train_data, train_masks_data = get_filenames(data)
        data = load_data_and_mask(train_data, train_masks_data)  # read into a data dict
        # loading image
        data["image"] = data["image"].astype(np.float32)  # shape (H W D)
        # loading labels
        class_shape = (1,) + data["image"].shape
        mask0 = np.zeros(class_shape)
        mask_list = []
        flagvect = np.ones((self.n_class,), np.float32)
        for i, mask in enumerate(data["label"]):
            if mask is None:
                mask = np.zeros(class_shape)
                flagvect[0] = 0
                flagvect[i + 1] = 0
            mask0 = np.logical_or(mask0, mask)
            mask_list.append(mask.reshape(class_shape))
        mask0 = 1 - mask0
        data["label"] = np.concatenate([mask0] + mask_list, axis=0).astype(np.uint8)  # shape (C H W D)
        # setting flags
        data["with_complete_groundtruth"] = flagvect  # flagvec is a boolean indicator for complete annotation
        return data

    def __len__(self):
        return len(self.data)


def train(n_feat, crop_size, bs, ep, optimizer="rmsprop", lr=5e-4, pretrain=None):
    model_name = f"./HaN_{n_feat}_{bs}_{ep}_{crop_size}_{lr}_"
    print(f"save the best model as '{model_name}' during training.")

    crop_size = [int(cz) for cz in crop_size.split(",")]
    print(f"input image crop_size: {crop_size}")

    # starting training set loader
    train_images = ImageLabelDataset(path=TRAIN_PATH, n_class=N_CLASSES)
    if np.any([cz == -1 for cz in crop_size]):  # using full image
        train_transform = Compose(
            [
                AddChannelDict(keys="image"),
                Rand3DElasticd(
                    keys=("image", "label"),
                    spatial_size=crop_size,
                    sigma_range=(10, 50),  # 30
                    magnitude_range=[600, 1200],  # 1000
                    prob=0.8,
                    rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                    shear_range=(np.pi / 18, np.pi / 18, np.pi / 18),
                    translate_range=(sz * 0.05 for sz in crop_size),
                    scale_range=(0.2, 0.2, 0.2),
                    mode=("bilinear", "nearest"),
                    padding_mode=("border", "zeros"),
                ),
            ]
        )
        train_dataset = Dataset(train_images, transform=train_transform)
        # when bs > 1, the loader assumes that the full image sizes are the same across the dataset
        train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=4, batch_size=bs, shuffle=True)
    else:
        # draw balanced foreground/background window samples according to the ground truth label
        train_transform = Compose(
            [
                AddChannelDict(keys="image"),
                SpatialPadd(keys=("image", "label"), spatial_size=crop_size),  # ensure image size >= crop_size
                RandCropByPosNegLabeld(
                    keys=("image", "label"), label_key="label", spatial_size=crop_size, num_samples=bs
                ),
                Rand3DElasticd(
                    keys=("image", "label"),
                    spatial_size=crop_size,
                    sigma_range=(10, 50),  # 30
                    magnitude_range=[600, 1200],  # 1000
                    prob=0.8,
                    rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                    shear_range=(np.pi / 18, np.pi / 18, np.pi / 18),
                    translate_range=(sz * 0.05 for sz in crop_size),
                    scale_range=(0.2, 0.2, 0.2),
                    mode=("bilinear", "nearest"),
                    padding_mode=("border", "zeros"),
                ),
            ]
        )
        train_dataset = Dataset(train_images, transform=train_transform)  # each dataset item is a list of windows
        train_dataloader = torch.utils.data.DataLoader(  # stack each dataset item into a single tensor
            train_dataset, num_workers=4, batch_size=1, shuffle=True, collate_fn=list_data_collate
        )
    first_sample = first(train_dataloader)
    print(first_sample["image"].shape)

    # starting validation set loader
    val_transform = Compose([AddChannelDict(keys="image")])
    val_dataset = Dataset(ImageLabelDataset(VAL_PATH, n_class=N_CLASSES), transform=val_transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=1, batch_size=1)
    print(val_dataset[0]["image"].shape)
    print(f"training images: {len(train_dataloader)}, validation images: {len(val_dataloader)}")

    model = UNetPipe(spatial_dims=3, in_channels=1, out_channels=N_CLASSES, n_feat=n_feat)
    model = flatten_sequential(model)
    lossweight = torch.from_numpy(np.array([2.22, 1.31, 1.99, 1.13, 1.93, 1.93, 1.0, 1.0, 1.90, 1.98], np.float32))

    if optimizer.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)  # lr = 5e-4
    elif optimizer.lower() == "momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # lr = 1e-4 for finetuning
    else:
        raise ValueError(f"Unknown optimizer type {optimizer}. (options are 'rmsprop' and 'momentum').")

    # config GPipe
    x = first_sample["image"].float()
    x = torch.autograd.Variable(x.cuda())
    partitions = torch.cuda.device_count()
    print(f"partition: {partitions}, input: {x.size()}")
    balance = balance_by_size(partitions, model, x)
    model = GPipe(model, balance, chunks=4, checkpoint="always")

    # config loss functions
    dice_loss_func = DiceLoss(softmax=True, reduction="none")
    # use the same pipeline and loss in
    # AnatomyNet: Deep learning for fast and fully automated whole‐volume segmentation of head and neck anatomy,
    # Medical Physics, 2018.
    focal_loss_func = FocalLoss(reduction="none")

    if pretrain:
        print(f"loading from {pretrain}.")
        pretrained_dict = torch.load(pretrain)["weight"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)

    b_time = time.time()
    best_val_loss = [0] * (N_CLASSES - 1)  # foreground
    best_ave = -1
    for epoch in range(ep):
        model.train()
        trainloss = 0
        for b_idx, data_dict in enumerate(train_dataloader):
            x_train = data_dict["image"]
            y_train = data_dict["label"]
            flagvec = data_dict["with_complete_groundtruth"]

            x_train = torch.autograd.Variable(x_train.cuda())
            y_train = torch.autograd.Variable(y_train.cuda().float())
            optimizer.zero_grad()
            o = model(x_train).to(0, non_blocking=True).float()

            loss = (dice_loss_func(o, y_train.to(o)) * flagvec.to(o) * lossweight.to(o)).mean()
            loss += 0.5 * (focal_loss_func(o, y_train.to(o)) * flagvec.to(o) * lossweight.to(o)).mean()
            loss.backward()
            optimizer.step()
            trainloss += loss.item()

            if b_idx % 20 == 0:
                print(f"Train Epoch: {epoch} [{b_idx}/{len(train_dataloader)}] \tLoss: {loss.item()}")
        print(f"epoch {epoch} TRAIN loss {trainloss / len(train_dataloader)}")

        if epoch % 10 == 0:
            model.eval()
            # check validation dice
            val_loss = [0] * (N_CLASSES - 1)
            n_val = [0] * (N_CLASSES - 1)
            for data_dict in val_dataloader:
                x_val = data_dict["image"]
                y_val = data_dict["label"]
                with torch.no_grad():
                    x_val = torch.autograd.Variable(x_val.cuda())
                o = model(x_val).to(0, non_blocking=True)
                loss = compute_meandice(o, y_val.to(o), mutually_exclusive=True, include_background=False)
                val_loss = [l.item() + tl if l == l else tl for l, tl in zip(loss[0], val_loss)]
                n_val = [n + 1 if l == l else n for l, n in zip(loss[0], n_val)]
            val_loss = [l / n for l, n in zip(val_loss, n_val)]
            print("validation scores %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % tuple(val_loss))
            for c in range(1, 10):
                if best_val_loss[c - 1] < val_loss[c - 1]:
                    best_val_loss[c - 1] = val_loss[c - 1]
                    state = {"epoch": epoch, "weight": model.state_dict(), "score_" + str(c): best_val_loss[c - 1]}
                    torch.save(state, f"{model_name}" + str(c))
            print("best validation scores %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % tuple(best_val_loss))

    print("total time", time.time() - b_time)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_feat", type=int, default=32, dest="n_feat")
    parser.add_argument("--crop_size", type=str, default="-1,-1,-1", dest="crop_size")
    parser.add_argument("--bs", type=int, default=1, dest="bs")  # batch size
    parser.add_argument("--ep", type=int, default=150, dest="ep")  # number of epochs
    parser.add_argument("--lr", type=float, default=5e-4, dest="lr")  # learning rate
    parser.add_argument("--optimizer", type=str, default="rmsprop", dest="optimizer")  # type of optimizer
    parser.add_argument("--pretrain", type=str, default=None, dest="pretrain")
    args = parser.parse_args()

    input_dict = vars(args)
    print(input_dict)
    train(**input_dict)

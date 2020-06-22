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

import numpy as np
import torch
from monai.transforms import SpatialPad, AddChannelDict, Compose
from monai.losses import MaskedDiceLoss
from monai.metrics import compute_meandice
from monai.data import Dataset
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_size

# from utils import pad, random_crop_pos_neg_multi_channel_3d
# from utils import focal, caldice
# from utils import tversky_loss_wmask_stable as tversky_loss_wmask
from unet_pipe import UNet

N_CLASSES = 10
TEST_PATH = "./testpddca15_crp.pth"  # path for processed data
PET_PATH = "./trainpddca15_pet_crp.pth"

torch.backends.cudnn.enabled = True


class ImageLabelDataset:
    def __init__(self, path, n_class=10):
        self.data = torch.load(path)
        self.n_class = n_class

    def __getitem__(self, index):
        data = self.data[index]
        img = data["img"].numpy().astype(np.float32)
        mask0 = np.zeros((1,) + img.shape)
        mask_list = []
        flagvect = np.ones((self.n_class,), np.float32)
        for i, mask in enumerate(data["mask"]):
            if mask is None:
                mask = np.zeros((1,) + img.shape)
                flagvect[0] = 0
                flagvect[i + 1] = 0
            mask0 = np.logical_or(mask0, mask)
            mask_list.append(mask.reshape((1,) + img.shape))
        mask0 = 1 - mask0
        label = np.concatenate([mask0] + mask_list, axis=0).astype(np.uint8)
        return {
            "image": img,  # shape (H, W, D)
            "label": label,  # shape (chns, H, W, D)
            "with_complete_groundtruth": flagvect,  # flagvec is a boolean indicator for complete annotation
        }

    def __len__(self):
        return len(self.data)


def train(n_feat, crop_size, bs, ep, pretrain=None):
    model = UNet(spatial_dims=3, in_channels=1, out_channels=N_CLASSES, n_feat=n_feat)
    if torch.cuda.is_available():
        model = model.cuda()
    lossweight = torch.from_numpy(np.array([2.22, 1.31, 1.99, 1.13, 1.93, 1.93, 1.0, 1.0, 1.90, 1.98], np.float32))

    crop_size = [int(cz) for cz in crop_size.split(",")]
    print(f"input image crop_size: {crop_size}")

    train_transform = Compose([AddChannelDict(keys="image")])
    traindataset = Dataset(ImageLabelDataset(path=PET_PATH, n_class=N_CLASSES), transform=train_transform)
    traindataloader = torch.utils.data.DataLoader(traindataset, num_workers=6, batch_size=bs, shuffle=True)

    # testdataset = Dataset(ImageLabelDataset(TEST_PATH, n_class=N_CLASSES))
    # testdataloader = torch.utils.data.DataLoader(testdataset, num_workers=6, batch_size=1)
    # print(f"training images: {len(traindataloader)}, validation images: {len(testdataloader)}")

    optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-4)
    data_dict = traindataset[0]
    x = data_dict["image"]
    x = torch.from_numpy(np.expand_dims(x, 0)).float()  # adds a batch dim
    x = torch.autograd.Variable(x.cuda())
    partitions = torch.cuda.device_count()
    print(partitions, x.size())
    balance = balance_by_size(partitions, model, x)
    model = GPipe(model, balance, chunks=4, checkpoint="always")
    loss = MaskedDiceLoss(softmax=True, reduction="none")

    if pretrain:
        pretrained_dict = torch.load(pretrain)["weight"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)

    b_time = time.time()
    for epoch in range(ep):
        model.train()
        trainloss = 0
        for data_dict in traindataloader:
            x_train = data_dict["image"]
            y_train = data_dict["label"]
            flagvec = data_dict["with_complete_groundtruth"]

            x_train = torch.autograd.Variable(x_train.cuda())
            y_train = torch.autograd.Variable(y_train.cuda().float())
            optimizer.zero_grad()
            o = model(x_train).to(0, non_blocking=True).float()

            # loss = tversky_loss_wmask(o, y_train, flagvec * torch.from_numpy(lossweight))
            # loss += 0.1 * focal(o, y_train, flagvec * torch.from_numpy(lossweight))
            loss = (loss(o, y_train) * flagvec * lossweight).mean()
            loss.backward()
            optimizer.step()
            trainloss += loss.item()
        print("epoch %i TRAIN loss %.4f" % (epoch, trainloss / len(traindataloader)))

        if epoch % 10 == 0:
            model.eval()
            # check training dice
            testloss = [0 for _ in range(N_CLASSES - 1)]
            ntest = [0 for _ in range(N_CLASSES - 1)]
            for data_dict in traindataloader:
                x_test = data_dict["image"]
                y_test = data_dict["label"]
                with torch.no_grad():
                    x_test = torch.autograd.Variable(x_test.cuda())
                o = model(x_test).to(0, non_blocking=True)
                loss = compute_meandice(o, y_test, mutually_exclusive=True, include_background=False)
                testloss = [l + tl if l != -1 else tl for l, tl in zip(loss, testloss)]
                ntest = [n + 1 if l != -1 else n for l, n in zip(loss, ntest)]
            testloss = [l / n for l, n in zip(testloss, ntest)]
            print("train scores %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % tuple(testloss))

    print("total time", time.time() - b_time)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_feat", type=int, default=32, dest="n_feat")
    parser.add_argument("--crop_size", type=str, default="-1,-1,-1", dest="crop_size")
    parser.add_argument("--bs", type=int, default=1, dest="bs")  # batch size
    parser.add_argument("--ep", type=int, default=150, dest="ep")  # 150
    parser.add_argument("--pretrain", type=str, default=None, dest="pretrain")
    args = parser.parse_args()

    train(**vars(args))

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
import shutil
import sys
import tempfile
import unittest
from glob import glob

import nibabel as nib
import numpy as np
import torch
from ignite.metrics import Accuracy

import monai
from monai.data import create_test_image_3d
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointLoader,
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    SegmentationSaver,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsChannelFirstd,
    AsDiscreted,
    Compose,
    KeepLargestConnectedComponentd,
    LoadNiftid,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    ToTensord,
)
from monai.utils import set_determinism
from tests.utils import skip_if_quick


def run_training_test(root_dir, device=torch.device("cuda:0"), amp=False):
    images = sorted(glob(os.path.join(root_dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(root_dir, "seg*.nii.gz")))
    train_files = [{"image": img, "label": seg} for img, seg in zip(images[:20], segs[:20])]
    val_files = [{"image": img, "label": seg} for img, seg in zip(images[-20:], segs[-20:])]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadNiftid(keys=["image", "label"]),
            AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
            ScaleIntensityd(keys=["image", "label"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadNiftid(keys=["image", "label"]),
            AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
            ScaleIntensityd(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # create a training data loader
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = monai.data.DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    # create a validation data loader
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4)

    # create UNet, DiceLoss and Adam optimizer
    net = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss = monai.losses.DiceLoss(sigmoid=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)

    val_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
        ]
    )
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=root_dir, output_transform=lambda x: None),
        TensorBoardImageHandler(
            log_dir=root_dir, batch_transform=lambda x: (x["image"], x["label"]), output_transform=lambda x: x["pred"]
        ),
        CheckpointSaver(save_dir=root_dir, save_dict={"net": net}, save_key_metric=True),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        post_transform=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=True, output_transform=lambda x: (x["pred"], x["label"]))
        },
        additional_metrics={"val_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))},
        val_handlers=val_handlers,
        amp=True if amp else False,
    )

    train_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
        ]
    )
    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=2, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
        TensorBoardStatsHandler(log_dir=root_dir, tag_name="train_loss", output_transform=lambda x: x["loss"]),
        CheckpointSaver(save_dir=root_dir, save_dict={"net": net, "opt": opt}, save_interval=2, epoch_level=True),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=5,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        inferer=SimpleInferer(),
        post_transform=train_post_transforms,
        key_train_metric={"train_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))},
        train_handlers=train_handlers,
        amp=True if amp else False,
    )
    trainer.run()

    return evaluator.state.best_metric


def run_inference_test(root_dir, model_file, device=torch.device("cuda:0"), amp=False):
    images = sorted(glob(os.path.join(root_dir, "im*.nii.gz")))
    segs = sorted(glob(os.path.join(root_dir, "seg*.nii.gz")))
    val_files = [{"image": img, "label": seg} for img, seg in zip(images, segs)]

    # define transforms for image and segmentation
    val_transforms = Compose(
        [
            LoadNiftid(keys=["image", "label"]),
            AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
            ScaleIntensityd(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4)

    # create UNet, DiceLoss and Adam optimizer
    net = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    val_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold_values=True),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
        ]
    )
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        CheckpointLoader(load_path=f"{model_file}", load_dict={"net": net}),
        SegmentationSaver(
            output_dir=root_dir,
            batch_transform=lambda batch: batch["image_meta_dict"],
            output_transform=lambda output: output["pred"],
        ),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        post_transform=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=True, output_transform=lambda x: (x["pred"], x["label"]))
        },
        additional_metrics={"val_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))},
        val_handlers=val_handlers,
        amp=True if amp else False,
    )
    evaluator.run()

    return evaluator.state.best_metric


class IntegrationWorkflows(unittest.TestCase):
    def setUp(self):
        set_determinism(seed=0)

        self.data_dir = tempfile.mkdtemp()
        for i in range(40):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"seg{i:d}.nii.gz"))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        monai.config.print_config()
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def tearDown(self):
        set_determinism(seed=None)
        shutil.rmtree(self.data_dir)

    @skip_if_quick
    def test_training(self):
        repeated = []
        test_rounds = 3 if monai.config.get_torch_version_tuple() >= (1, 6) else 2
        for i in range(test_rounds):
            set_determinism(seed=0)

            repeated.append([])
            best_metric = run_training_test(self.data_dir, device=self.device, amp=(i == 2))
            print("best metric", best_metric)
            if i == 2:
                np.testing.assert_allclose(best_metric, 0.9219996750354766, rtol=1e-2)
            else:
                np.testing.assert_allclose(best_metric, 0.921965891122818, rtol=1e-2)
            repeated[i].append(best_metric)

            model_file = sorted(glob(os.path.join(self.data_dir, "net_key_metric*.pt")))[-1]
            infer_metric = run_inference_test(self.data_dir, model_file, device=self.device, amp=(i == 2))
            print("infer metric", infer_metric)
            # check inference properties
            if i == 2:
                np.testing.assert_allclose(infer_metric, 0.9217855930328369, rtol=1e-2)
            else:
                np.testing.assert_allclose(infer_metric, 0.9217526227235794, rtol=1e-2)
            repeated[i].append(infer_metric)

            output_files = sorted(glob(os.path.join(self.data_dir, "img*", "*.nii.gz")))
            if i == 2:
                sums = [
                    0.14183807373046875,
                    0.15151405334472656,
                    0.13811445236206055,
                    0.1336650848388672,
                    0.1842341423034668,
                    0.16353750228881836,
                    0.14104795455932617,
                    0.16643333435058594,
                    0.15668964385986328,
                    0.1764383316040039,
                    0.16112232208251953,
                    0.1641840934753418,
                    0.14401578903198242,
                    0.11075973510742188,
                    0.16075706481933594,
                    0.19603967666625977,
                    0.1743607521057129,
                    0.05361223220825195,
                    0.19009971618652344,
                    0.19875097274780273,
                    0.19498729705810547,
                    0.2027440071105957,
                    0.16035127639770508,
                    0.13188838958740234,
                    0.15143728256225586,
                    0.1370086669921875,
                    0.22630071640014648,
                    0.16111421585083008,
                    0.14713764190673828,
                    0.10443782806396484,
                    0.11977195739746094,
                    0.13068008422851562,
                    0.11225223541259766,
                    0.15175437927246094,
                    0.1594991683959961,
                    0.1894702911376953,
                    0.21605825424194336,
                    0.17748403549194336,
                    0.18474626541137695,
                    0.03627157211303711,
                ]
            else:
                sums = [
                    0.14183568954467773,
                    0.15139484405517578,
                    0.13803958892822266,
                    0.13356733322143555,
                    0.18455982208251953,
                    0.16363763809204102,
                    0.14090299606323242,
                    0.16649341583251953,
                    0.15651702880859375,
                    0.17655181884765625,
                    0.1611647605895996,
                    0.1644759178161621,
                    0.14383649826049805,
                    0.11055231094360352,
                    0.16080236434936523,
                    0.19629907608032227,
                    0.17441368103027344,
                    0.053577423095703125,
                    0.19043731689453125,
                    0.19904851913452148,
                    0.19525957107543945,
                    0.20304203033447266,
                    0.16030073165893555,
                    0.13170528411865234,
                    0.15118885040283203,
                    0.13686418533325195,
                    0.22668886184692383,
                    0.1611466407775879,
                    0.1472468376159668,
                    0.10427331924438477,
                    0.11962461471557617,
                    0.1305699348449707,
                    0.11204767227172852,
                    0.15171241760253906,
                    0.1596231460571289,
                    0.18976259231567383,
                    0.21649408340454102,
                    0.17761707305908203,
                    0.1851673126220703,
                    0.036365509033203125,
                ]
            for (output, s) in zip(output_files, sums):
                ave = np.mean(nib.load(output).get_fdata())
                np.testing.assert_allclose(ave, s, rtol=1e-2)
                repeated[i].append(ave)
        np.testing.assert_allclose(repeated[0], repeated[1])


if __name__ == "__main__":
    unittest.main()

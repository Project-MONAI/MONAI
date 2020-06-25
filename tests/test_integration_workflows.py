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

import os
import sys
import shutil
import tempfile
import unittest
from glob import glob
import logging
import nibabel as nib
import numpy as np
import torch
from ignite.metrics import Accuracy

import monai
from monai.transforms import (
    Compose,
    LoadNiftid,
    AsChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ToTensord,
    Activationsd,
    AsDiscreted,
    KeepLargestConnectedComponentd,
)
from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    TensorBoardImageHandler,
    ValidationHandler,
    LrScheduleHandler,
    CheckpointSaver,
    CheckpointLoader,
    SegmentationSaver,
    MeanDice,
)
from monai.data import create_test_image_3d
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.utils import set_determinism
from tests.utils import skip_if_quick


def run_training_test(root_dir, device=torch.device("cuda:0")):
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
        amp=False,
        post_transform=train_post_transforms,
        key_train_metric={"train_acc": Accuracy(output_transform=lambda x: (x["pred"], x["label"]))},
        train_handlers=train_handlers,
    )
    trainer.run()

    return evaluator.state.best_metric


def run_inference_test(root_dir, model_file, device=torch.device("cuda:0")):
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
        for i in range(2):
            torch.manual_seed(0)

            repeated.append([])
            best_metric = run_training_test(self.data_dir, device=self.device)
            print("best metric", best_metric)
            np.testing.assert_allclose(best_metric, 0.9232678800821305, rtol=1e-3)
            repeated[i].append(best_metric)

            model_file = sorted(glob(os.path.join(self.data_dir, "net_key_metric*.pth")))[-1]
            infer_metric = run_inference_test(self.data_dir, model_file, device=self.device)
            print("infer metric", infer_metric)
            # check inference properties
            np.testing.assert_allclose(infer_metric, 0.9224808603525162, rtol=1e-3)
            repeated[i].append(infer_metric)
            output_files = sorted(glob(os.path.join(self.data_dir, "img*", "*.nii.gz")))
            sums = [
                0.14212512969970703,
                0.1506481170654297,
                0.1368846893310547,
                0.13330554962158203,
                0.18573999404907227,
                0.1647019386291504,
                0.1408066749572754,
                0.16658973693847656,
                0.15639686584472656,
                0.17746448516845703,
                0.16197776794433594,
                0.16469907760620117,
                0.14304876327514648,
                0.10998392105102539,
                0.16064167022705078,
                0.1962604522705078,
                0.17453575134277344,
                0.052756309509277344,
                0.19060277938842773,
                0.20035600662231445,
                0.19619369506835938,
                0.20325279235839844,
                0.15996408462524414,
                0.13104581832885742,
                0.14955568313598633,
                0.135528564453125,
                0.2252669334411621,
                0.16170835494995117,
                0.14747190475463867,
                0.10289239883422852,
                0.11845922470092773,
                0.13117074966430664,
                0.11201333999633789,
                0.15172672271728516,
                0.15926742553710938,
                0.18946075439453125,
                0.21686124801635742,
                0.1773381233215332,
                0.1864323616027832,
                0.035613059997558594,
            ]
            for (output, s) in zip(output_files, sums):
                ave = np.mean(nib.load(output).get_fdata())
                np.testing.assert_allclose(ave, s, rtol=1e-2)
                repeated[i].append(ave)
        np.testing.assert_allclose(repeated[0], repeated[1])


if __name__ == "__main__":
    unittest.main()

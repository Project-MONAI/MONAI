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
                np.testing.assert_allclose(best_metric, 0.9218417823314666, rtol=1e-2)
            else:
                np.testing.assert_allclose(best_metric, 0.9219249188899994, rtol=1e-2)
            repeated[i].append(best_metric)

            model_file = sorted(glob(os.path.join(self.data_dir, "net_key_metric*.pt")))[-1]
            infer_metric = run_inference_test(self.data_dir, model_file, device=self.device, amp=(i == 2))
            print("infer metric", infer_metric)
            # check inference properties
            if i == 2:
                np.testing.assert_allclose(infer_metric, 0.9216801583766937, rtol=1e-2)
            else:
                np.testing.assert_allclose(infer_metric, 0.9217007756233215, rtol=1e-2)
            repeated[i].append(infer_metric)

            output_files = sorted(glob(os.path.join(self.data_dir, "img*", "*.nii.gz")))
            if i == 2:
                sums = [
                    0.14187908172607422,
                    0.15141582489013672,
                    0.13805580139160156,
                    0.1335921287536621,
                    0.18461370468139648,
                    0.1636824607849121,
                    0.1409168243408203,
                    0.1665506362915039,
                    0.15652847290039062,
                    0.17659997940063477,
                    0.16120052337646484,
                    0.1645350456237793,
                    0.14385366439819336,
                    0.11049985885620117,
                    0.16086244583129883,
                    0.19636201858520508,
                    0.17445993423461914,
                    0.05356025695800781,
                    0.19049406051635742,
                    0.19910669326782227,
                    0.1953139305114746,
                    0.2030935287475586,
                    0.1603412628173828,
                    0.1317133903503418,
                    0.1511821746826172,
                    0.13686084747314453,
                    0.22674274444580078,
                    0.16118431091308594,
                    0.14728116989135742,
                    0.10426139831542969,
                    0.11961984634399414,
                    0.13056421279907227,
                    0.11201953887939453,
                    0.15174198150634766,
                    0.15967845916748047,
                    0.1898040771484375,
                    0.2165522575378418,
                    0.17767810821533203,
                    0.18523073196411133,
                    0.03636026382446289,
                ]
            else:
                sums = [
                    0.14187145233154297,
                    0.15141057968139648,
                    0.13805103302001953,
                    0.13358211517333984,
                    0.18460750579833984,
                    0.16367673873901367,
                    0.14090681076049805,
                    0.16654396057128906,
                    0.1565251350402832,
                    0.17658662796020508,
                    0.16119146347045898,
                    0.1645212173461914,
                    0.1438441276550293,
                    0.11049842834472656,
                    0.1608572006225586,
                    0.1963505744934082,
                    0.17445039749145508,
                    0.0535578727722168,
                    0.19048500061035156,
                    0.19910240173339844,
                    0.19530916213989258,
                    0.20308685302734375,
                    0.16033363342285156,
                    0.13170766830444336,
                    0.15117835998535156,
                    0.13685131072998047,
                    0.22673511505126953,
                    0.16117477416992188,
                    0.14727354049682617,
                    0.10425710678100586,
                    0.11961698532104492,
                    0.13056182861328125,
                    0.11200904846191406,
                    0.1517343521118164,
                    0.15967321395874023,
                    0.18979740142822266,
                    0.21654462814331055,
                    0.17766666412353516,
                    0.18522214889526367,
                    0.03636026382446289,
                ]
            for (output, s) in zip(output_files, sums):
                ave = np.mean(nib.load(output).get_fdata())
                np.testing.assert_allclose(ave, s, rtol=1e-2)
                repeated[i].append(ave)
        np.testing.assert_allclose(repeated[0], repeated[1])


if __name__ == "__main__":
    unittest.main()

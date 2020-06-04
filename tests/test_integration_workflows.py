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
                keys=["image", "label"], label_key="label", size=[96, 96, 96], pos=1, neg=1, num_samples=4
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
    loss = monai.losses.DiceLoss(do_sigmoid=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)

    val_post_transforms = Compose(
        [
            Activationsd(keys="pred", output_postfix="act", sigmoid=True),
            AsDiscreted(keys="pred_act", output_postfix="dis", threshold_values=True),
            KeepLargestConnectedComponentd(keys="pred_act_dis", applied_values=[1], output_postfix=None),
        ]
    )
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=root_dir, output_transform=lambda x: None),
        TensorBoardImageHandler(
            log_dir=root_dir,
            batch_transform=lambda x: (x["image"], x["label"]),
            output_transform=lambda x: x["pred_act_dis"],
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
            "val_mean_dice": MeanDice(
                include_background=True, output_transform=lambda x: (x["pred_act_dis"], x["label"])
            )
        },
        additional_metrics={"val_acc": Accuracy(output_transform=lambda x: (x["pred_act_dis"], x["label"]))},
        val_handlers=val_handlers,
    )

    train_post_transforms = Compose(
        [
            Activationsd(keys="pred", output_postfix="act", sigmoid=True),
            AsDiscreted(keys="pred_act", output_postfix="dis", threshold_values=True),
            KeepLargestConnectedComponentd(keys="pred_act_dis", applied_values=[1], output_postfix=None),
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
        key_train_metric={"train_acc": Accuracy(output_transform=lambda x: (x["pred_act_dis"], x["label"]))},
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
            Activationsd(keys="pred", output_postfix="act", sigmoid=True),
            AsDiscreted(keys="pred_act", output_postfix="dis", threshold_values=True),
            KeepLargestConnectedComponentd(keys="pred_act_dis", applied_values=[1], output_postfix=None),
        ]
    )
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        CheckpointLoader(load_path=f"{model_file}", load_dict={"net": net}),
        SegmentationSaver(
            output_dir=root_dir,
            batch_transform=lambda x: {
                "filename_or_obj": x["image.filename_or_obj"],
                "affine": x["image.affine"],
                "original_affine": x["image.original_affine"],
                "spatial_shape": x["image.spatial_shape"],
            },
            output_transform=lambda x: x["pred_act_dis"],
        ),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        post_transform=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=True, output_transform=lambda x: (x["pred_act_dis"], x["label"])
            )
        },
        additional_metrics={"val_acc": Accuracy(output_transform=lambda x: (x["pred_act_dis"], x["label"]))},
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
            np.testing.assert_allclose(best_metric, 0.9233065992593765, rtol=1e-3)
            repeated[i].append(best_metric)
            model_file = sorted(glob(os.path.join(self.data_dir, "net_key_metric*.pth")))[-1]
            infer_metric = run_inference_test(self.data_dir, model_file, device=self.device)
            print("infer metric", infer_metric)
            # check inference properties
            np.testing.assert_allclose(infer_metric, 0.9225203782320023, rtol=1e-3)
            repeated[i].append(infer_metric)
            output_files = sorted(glob(os.path.join(self.data_dir, "img*", "*.nii.gz")))
            sums = [
                0.14210844039916992,
                0.15063095092773438,
                0.1368732452392578,
                0.13329029083251953,
                0.18572759628295898,
                0.16469240188598633,
                0.14079713821411133,
                0.16657590866088867,
                0.15638303756713867,
                0.1774449348449707,
                0.16196775436401367,
                0.1646862030029297,
                0.14303207397460938,
                0.10997390747070312,
                0.1606287956237793,
                0.19624853134155273,
                0.17452192306518555,
                0.052750587463378906,
                0.19059038162231445,
                0.20033836364746094,
                0.19617938995361328,
                0.20324134826660156,
                0.15994548797607422,
                0.13103246688842773,
                0.14954280853271484,
                0.13551807403564453,
                0.2252488136291504,
                0.16169977188110352,
                0.14746522903442383,
                0.10287714004516602,
                0.11845254898071289,
                0.13115501403808594,
                0.11200284957885742,
                0.15171051025390625,
                0.1592564582824707,
                0.18944692611694336,
                0.21684503555297852,
                0.1773233413696289,
                0.18641948699951172,
                0.03560972213745117
            ]
            for (output, s) in zip(output_files, sums):
                ave = np.mean(nib.load(output).get_fdata())
                print(ave, ",")
            for (output, s) in zip(output_files, sums):
                ave = np.mean(nib.load(output).get_fdata())
                np.testing.assert_allclose(ave, s, rtol=1e-3)
                repeated[i].append(ave)
        np.testing.assert_allclose(repeated[0], repeated[1])


if __name__ == "__main__":
    unittest.main()

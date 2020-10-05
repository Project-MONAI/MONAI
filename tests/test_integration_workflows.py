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
                np.testing.assert_allclose(best_metric, 0.921953222155571, rtol=1e-2)
            else:
                np.testing.assert_allclose(best_metric, 0.9222842544317246, rtol=1e-2)
            repeated[i].append(best_metric)

            model_file = sorted(glob(os.path.join(self.data_dir, "net_key_metric*.pt")))[-1]
            infer_metric = run_inference_test(self.data_dir, model_file, device=self.device, amp=(i == 2))
            print("infer metric", infer_metric)
            # check inference properties
            if i == 2:
                np.testing.assert_allclose(infer_metric, 0.9218754783272743, rtol=1e-2)
            else:
                np.testing.assert_allclose(infer_metric, 0.9218612298369407, rtol=1e-2)
            repeated[i].append(infer_metric)

            output_files = sorted(glob(os.path.join(self.data_dir, "img*", "*.nii.gz")))
            if i == 2:
                sums = [
                    0.14185380935668945,
                    0.1513509750366211,
                    0.13789939880371094,
                    0.1335468292236328,
                    0.18503952026367188,
                    0.16396522521972656,
                    0.14049530029296875,
                    0.1668844223022461,
                    0.15607881546020508,
                    0.1767129898071289,
                    0.16128301620483398,
                    0.16508150100708008,
                    0.14340972900390625,
                    0.1096959114074707,
                    0.16115188598632812,
                    0.19690465927124023,
                    0.1747732162475586,
                    0.052988529205322266,
                    0.19108295440673828,
                    0.1993880271911621,
                    0.19587039947509766,
                    0.20384740829467773,
                    0.16034507751464844,
                    0.1313328742980957,
                    0.15056657791137695,
                    0.1366105079650879,
                    0.22742700576782227,
                    0.1611924171447754,
                    0.14752864837646484,
                    0.10315418243408203,
                    0.11873531341552734,
                    0.13009166717529297,
                    0.11156511306762695,
                    0.15167236328125,
                    0.16005373001098633,
                    0.1902608871459961,
                    0.21702146530151367,
                    0.17800235748291016,
                    0.18591070175170898,
                    0.03580522537231445,
                ]
            else:
                sums = [
                    0.1418623924255371,
                    0.15135431289672852,
                    0.13790512084960938,
                    0.13355112075805664,
                    0.1850442886352539,
                    0.163970947265625,
                    0.1405010223388672,
                    0.16689109802246094,
                    0.15608596801757812,
                    0.17671585083007812,
                    0.1612858772277832,
                    0.1650848388671875,
                    0.14341402053833008,
                    0.10969829559326172,
                    0.16115999221801758,
                    0.19691038131713867,
                    0.17478561401367188,
                    0.052988529205322266,
                    0.1910867691040039,
                    0.19939565658569336,
                    0.1958761215209961,
                    0.2038559913635254,
                    0.16034746170043945,
                    0.1313328742980957,
                    0.15057611465454102,
                    0.13661479949951172,
                    0.22742986679077148,
                    0.16119623184204102,
                    0.14753484725952148,
                    0.10316228866577148,
                    0.11874008178710938,
                    0.13009357452392578,
                    0.11156797409057617,
                    0.15167999267578125,
                    0.16005420684814453,
                    0.19026756286621094,
                    0.2170271873474121,
                    0.17800474166870117,
                    0.1859145164489746,
                    0.03580617904663086,
                ]
            for (output, s) in zip(output_files, sums):
                ave = np.mean(nib.load(output).get_fdata())
                np.testing.assert_allclose(ave, s, rtol=1e-2)
                repeated[i].append(ave)
        np.testing.assert_allclose(repeated[0], repeated[1])


if __name__ == "__main__":
    unittest.main()

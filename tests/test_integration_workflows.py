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
from tests.utils import skip_if_quick, get_expected

EXPECTED = {
    "1.6.0": {
        "best_metric": 0.9219646483659745,
        "infer_metric": 0.921751058101654,
        "output_sums": [
            0.14183664321899414,
            0.1513957977294922,
            0.13804054260253906,
            0.13356828689575195,
            0.18456125259399414,
            0.16363763809204102,
            0.14090299606323242,
            0.16649389266967773,
            0.15651893615722656,
            0.17655134201049805,
            0.16116666793823242,
            0.1644763946533203,
            0.14383649826049805,
            0.11055326461791992,
            0.16080379486083984,
            0.19629907608032227,
            0.17441415786743164,
            0.053577423095703125,
            0.19043779373168945,
            0.19904804229736328,
            0.19526052474975586,
            0.20304107666015625,
            0.16030025482177734,
            0.13170623779296875,
            0.15118932723999023,
            0.13686418533325195,
            0.22668886184692383,
            0.1611471176147461,
            0.1472463607788086,
            0.10427379608154297,
            0.11962461471557617,
            0.1305704116821289,
            0.11204910278320312,
            0.15171337127685547,
            0.15962505340576172,
            0.18976259231567383,
            0.21649456024169922,
            0.17761802673339844,
            0.18516874313354492,
            0.03636503219604492,
        ],
        "best_metric_2": 0.9219559609889985,
        "infer_metric_2": 0.9217371672391892,
        "output_sums_2": [
            0.14187288284301758,
            0.15140819549560547,
            0.13802719116210938,
            0.1335887908935547,
            0.18454980850219727,
            0.1636652946472168,
            0.14091157913208008,
            0.16653108596801758,
            0.15651702880859375,
            0.17658615112304688,
            0.1611957550048828,
            0.16448307037353516,
            0.14385128021240234,
            0.1105203628540039,
            0.16085100173950195,
            0.19626951217651367,
            0.17442035675048828,
            0.053586483001708984,
            0.19042730331420898,
            0.1990523338317871,
            0.1952815055847168,
            0.20303773880004883,
            0.16034317016601562,
            0.13172531127929688,
            0.15118741989135742,
            0.1368694305419922,
            0.22667837142944336,
            0.16119050979614258,
            0.14726591110229492,
            0.10426473617553711,
            0.11961841583251953,
            0.13054800033569336,
            0.11203193664550781,
            0.15172529220581055,
            0.15963029861450195,
            0.18975019454956055,
            0.21646499633789062,
            0.17763566970825195,
            0.18517112731933594,
            0.03638744354248047,
        ],
    },
    "1.7.0": {
        "best_metric": 0.9217087924480438,
        "infer_metric": 0.9214379042387009,
        "output_sums": [
            0.14209461212158203,
            0.15126705169677734,
            0.13800382614135742,
            0.1338181495666504,
            0.1850571632385254,
            0.16372442245483398,
            0.14059066772460938,
            0.16674423217773438,
            0.15653657913208008,
            0.17690563201904297,
            0.16154909133911133,
            0.16521310806274414,
            0.14388608932495117,
            0.1103353500366211,
            0.1609959602355957,
            0.1967010498046875,
            0.1746964454650879,
            0.05329275131225586,
            0.19098854064941406,
            0.19976520538330078,
            0.19576644897460938,
            0.20346736907958984,
            0.1601848602294922,
            0.1316051483154297,
            0.1511220932006836,
            0.13670969009399414,
            0.2276287078857422,
            0.1611800193786621,
            0.14751672744750977,
            0.10413789749145508,
            0.11944007873535156,
            0.1305546760559082,
            0.11204719543457031,
            0.15145111083984375,
            0.16007614135742188,
            0.1904129981994629,
            0.21741962432861328,
            0.17812013626098633,
            0.18587207794189453,
            0.03605222702026367,
        ],
        "best_metric_2": 0.9210659921169281,
        "infer_metric_2": 0.9208109736442566,
        "output_sums_2": [
            0.14227628707885742,
            0.1515035629272461,
            0.13819408416748047,
            0.13402271270751953,
            0.18525266647338867,
            0.16388607025146484,
            0.14076614379882812,
            0.16694307327270508,
            0.15677356719970703,
            0.1771831512451172,
            0.16172313690185547,
            0.1653728485107422,
            0.14413118362426758,
            0.11057281494140625,
            0.16121912002563477,
            0.19680166244506836,
            0.1748638153076172,
            0.053426265716552734,
            0.19117307662963867,
            0.19996356964111328,
            0.1959366798400879,
            0.20363712310791016,
            0.16037797927856445,
            0.13180780410766602,
            0.1513657569885254,
            0.13686084747314453,
            0.2277364730834961,
            0.16137409210205078,
            0.1476879119873047,
            0.10438394546508789,
            0.11967992782592773,
            0.13080739974975586,
            0.11226606369018555,
            0.15168476104736328,
            0.1602616310119629,
            0.190582275390625,
            0.21756458282470703,
            0.17825984954833984,
            0.18604803085327148,
            0.036206722259521484,
        ],
    },
}


def run_training_test(root_dir, device="cuda:0", amp=False):
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


def run_inference_test(root_dir, model_file, device="cuda:0", amp=False):
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
                np.testing.assert_allclose(best_metric, get_expected(EXPECTED, key="best_metric_2"), rtol=1e-2)
            else:
                np.testing.assert_allclose(best_metric, get_expected(EXPECTED, key="best_metric"), rtol=1e-2)
            repeated[i].append(best_metric)

            model_file = sorted(glob(os.path.join(self.data_dir, "net_key_metric*.pt")))[-1]
            infer_metric = run_inference_test(self.data_dir, model_file, device=self.device, amp=(i == 2))
            print("infer metric", infer_metric)
            # check inference properties
            if i == 2:
                np.testing.assert_allclose(infer_metric, get_expected(EXPECTED, key="infer_metric_2"), rtol=1e-2)
            else:
                np.testing.assert_allclose(infer_metric, get_expected(EXPECTED, key="infer_metric"), rtol=1e-2)
            repeated[i].append(infer_metric)

            output_files = sorted(glob(os.path.join(self.data_dir, "img*", "*.nii.gz")))
            if i == 2:
                sums = get_expected(EXPECTED, key="output_sums_2")
            else:
                sums = get_expected(EXPECTED, key="output_sums")
            for (output, s) in zip(output_files, sums):
                ave = np.mean(nib.load(output).get_fdata())
                np.testing.assert_allclose(ave, s, rtol=1e-2)
                repeated[i].append(ave)
        np.testing.assert_allclose(repeated[0], repeated[1])


if __name__ == "__main__":
    unittest.main()

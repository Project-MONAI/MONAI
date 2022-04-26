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

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from monai.apps.deepedit.interaction import Interaction
from monai.apps.deepedit.transforms import (
    AddGuidanceSignalCustomd,
    AddInitialSeedPointMissingLabelsd,
    AddRandomGuidanceCustomd,
    FindAllValidSlicesMissingLabelsd,
    FindDiscrepancyRegionsCustomd,
    SplitPredsLabeld,
)
from monai.data import Dataset
from monai.engines import SupervisedTrainer
from monai.engines.utils import IterationEvents
from monai.losses import DiceCELoss
from monai.transforms import Activationsd, AsDiscreted, Compose, ToNumpyd, ToTensord


def add_one(engine):
    if engine.state.best_metric == -1:
        engine.state.best_metric = 0
    else:
        engine.state.best_metric = engine.state.best_metric + 1


class TestInteractions(unittest.TestCase):
    def run_interaction(self, train, compose):
        label_names = {"spleen": 1, "background": 0}
        np.random.seed(0)
        data = [
            {
                "image": np.random.randint(0, 256, size=(1, 10, 10, 10)).astype(np.float32),
                "label": np.random.randint(0, 2, size=(1, 10, 10, 10)),
                "label_names": label_names,
            }
            for _ in range(5)
        ]
        network = torch.nn.Conv3d(3, 2, 1)
        lr = 1e-3
        opt = torch.optim.SGD(network.parameters(), lr)
        loss = DiceCELoss(to_onehot_y=True, softmax=True)
        train_transforms = Compose(
            [
                FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
                AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
                AddGuidanceSignalCustomd(keys="image", guidance="guidance", number_intensity_ch=1),
                ToTensord(keys=("image", "label")),
            ]
        )
        dataset = Dataset(data, transform=train_transforms)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        iteration_transforms = [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(
                keys=("pred", "label"), argmax=(True, False), to_onehot=(True, True), n_classes=len(label_names)
            ),
            FindDiscrepancyRegionsCustomd(keys="label", pred="pred", discrepancy="discrepancy"),
            AddRandomGuidanceCustomd(
                keys="NA", guidance="guidance", discrepancy="discrepancy", probability="probability"
            ),
            AddGuidanceSignalCustomd(keys="image", guidance="guidance", number_intensity_ch=1),
            ToTensord(keys=("image", "label")),
        ]
        iteration_transforms = Compose(iteration_transforms) if compose else iteration_transforms

        # i = Interaction(transforms=iteration_transforms, train=train, max_interactions=5)
        i = Interaction(
            deepgrow_probability=1.0,
            transforms=iteration_transforms,
            click_probability_key="probability",
            train=False,
            label_names=label_names,
        )
        self.assertEqual(len(i.transforms.transforms), 6, "Mismatch in expected transforms")

        # set up engine
        engine = SupervisedTrainer(
            device=torch.device("cpu"),
            max_epochs=1,
            train_data_loader=data_loader,
            network=network,
            optimizer=opt,
            loss_function=loss,
            iteration_update=i,
        )
        engine.add_event_handler(IterationEvents.INNER_ITERATION_STARTED, add_one)
        engine.add_event_handler(IterationEvents.INNER_ITERATION_COMPLETED, add_one)

        engine.run()
        self.assertIsNotNone(engine.state.batch[0].get("guidance"), "guidance is missing")
        self.assertEqual(engine.state.best_metric, 9)

    def test_train_interaction(self):
        self.run_interaction(train=True, compose=True)

    def test_val_interaction(self):
        self.run_interaction(train=False, compose=False)


if __name__ == "__main__":
    unittest.main()

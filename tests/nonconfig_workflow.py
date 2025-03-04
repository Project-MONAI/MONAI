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

from __future__ import annotations

import torch

from monai.bundle import BundleWorkflow, PythonicWorkflow
from monai.data import DataLoader, Dataset
from monai.engines import SupervisedEvaluator
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    SaveImaged,
    ScaleIntensityd,
    ScaleIntensityRanged,
)
from monai.utils import BundleProperty, CommonKeys, set_determinism


class NonConfigWorkflow(BundleWorkflow):
    """
    Test class simulates the bundle workflow defined by Python script directly.

    """

    def __init__(self, filename, output_dir, meta_file=None, logging_file=None):
        super().__init__(workflow_type="inference", meta_file=meta_file, logging_file=logging_file)
        self.filename = filename
        self.output_dir = output_dir
        self._bundle_root = "will override"
        self._dataset_dir = "."
        self._device = torch.device("cpu")
        self._data = [{"image": self.filename}]
        self._dataset = None
        self._network_def = None
        self._inferer = None
        self._preprocessing = None
        self._postprocessing = None
        self._evaluator = None
        self._version = None
        self._monai_version = None
        self._pytorch_version = None
        self._numpy_version = None

    def initialize(self):
        set_determinism(0)
        if self._version is None:
            self._version = "0.1.0"

        if self._monai_version is None:
            self._monai_version = "1.1.0"

        if self._pytorch_version is None:
            self._pytorch_version = "2.3.0"

        if self._numpy_version is None:
            self._numpy_version = "1.22.2"

        if self._preprocessing is None:
            self._preprocessing = Compose(
                [LoadImaged(keys="image"), EnsureChannelFirstd(keys="image"), ScaleIntensityd(keys="image")]
            )
        self._dataset = Dataset(data=self._data, transform=self._preprocessing)
        dataloader = DataLoader(self._dataset, batch_size=1, num_workers=4)

        if self._network_def is None:
            self._network_def = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=[2, 2, 4, 8, 4],
                strides=[2, 2, 2, 2],
                num_res_units=2,
                norm="batch",
            )
        if self._inferer is None:
            self._inferer = SlidingWindowInferer(roi_size=(64, 64, 32), sw_batch_size=4, overlap=0.25)

        if self._postprocessing is None:
            self._postprocessing = Compose(
                [
                    Activationsd(keys="pred", softmax=True),
                    AsDiscreted(keys="pred", argmax=True),
                    SaveImaged(keys="pred", output_dir=self.output_dir, output_postfix="seg"),
                ]
            )

        self._evaluator = SupervisedEvaluator(
            device=self._device,
            val_data_loader=dataloader,
            network=self._network_def.to(self._device),
            inferer=self._inferer,
            postprocessing=self._postprocessing,
            amp=False,
        )

    def run(self):
        self._evaluator.run()

    def finalize(self):
        return True

    def _get_property(self, name, property):
        if name == "bundle_root":
            return self._bundle_root
        if name == "dataset_dir":
            return self._dataset_dir
        if name == "dataset_data":
            return self._data
        if name == "dataset":
            return self._dataset
        if name == "device":
            return self._device
        if name == "evaluator":
            return self._evaluator
        if name == "network_def":
            return self._network_def
        if name == "inferer":
            return self._inferer
        if name == "preprocessing":
            return self._preprocessing
        if name == "postprocessing":
            return self._postprocessing
        if name == "version":
            return self._version
        if name == "monai_version":
            return self._monai_version
        if name == "pytorch_version":
            return self._pytorch_version
        if name == "numpy_version":
            return self._numpy_version
        if property[BundleProperty.REQUIRED]:
            raise ValueError(f"unsupported property '{name}' is required in the bundle properties.")

    def _set_property(self, name, property, value):
        if name == "bundle_root":
            self._bundle_root = value
        elif name == "device":
            self._device = value
        elif name == "dataset_dir":
            self._dataset_dir = value
        elif name == "dataset_data":
            self._data = value
        elif name == "dataset":
            self._dataset = value
        elif name == "evaluator":
            self._evaluator = value
        elif name == "network_def":
            self._network_def = value
        elif name == "inferer":
            self._inferer = value
        elif name == "preprocessing":
            self._preprocessing = value
        elif name == "postprocessing":
            self._postprocessing = value
        elif name == "version":
            self._version = value
        elif name == "monai_version":
            self._monai_version = value
        elif name == "pytorch_version":
            self._pytorch_version = value
        elif name == "numpy_version":
            self._numpy_version = value
        elif property[BundleProperty.REQUIRED]:
            raise ValueError(f"unsupported property '{name}' is required in the bundle properties.")


class PythonicWorkflowImpl(PythonicWorkflow):
    """
    Test class simulates the bundle workflow defined by Python script directly.
    """

    def __init__(
        self,
        workflow_type: str = "inference",
        config_file: str | None = None,
        properties_path: str | None = None,
        meta_file: str | None = None,
    ):
        super().__init__(
            workflow_type=workflow_type, properties_path=properties_path, config_file=config_file, meta_file=meta_file
        )
        self.dataflow: dict = {}

    def initialize(self):
        self._props_vals = {}
        self._is_initialized = True
        self.net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        preprocessing = Compose(
            [
                EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityd(keys="image"),
                ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            ]
        )
        self.dataset = Dataset(data=[self.dataflow], transform=preprocessing)
        self.postprocessing = Compose([Activationsd(keys="pred", softmax=True), AsDiscreted(keys="pred", argmax=True)])

    def run(self):
        data = self.dataset[0]
        inputs = data[CommonKeys.IMAGE].unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            data[CommonKeys.PRED] = self.inferer(inputs, self.net)
        self.dataflow.update({CommonKeys.PRED: self.postprocessing(data)[CommonKeys.PRED]})

    def finalize(self):
        pass

    def get_bundle_root(self):
        return "."

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_inferer(self):
        return SlidingWindowInferer(roi_size=self.parser.roi_size, sw_batch_size=1, overlap=0)

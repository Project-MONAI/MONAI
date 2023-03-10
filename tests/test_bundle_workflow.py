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

import os
import shutil
import tempfile
import unittest
from copy import deepcopy

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized

from monai.bundle import BundleWorkflow, ConfigWorkflow
from monai.data import DataLoader, Dataset
from monai.engines import SupervisedEvaluator
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    SaveImaged,
    ScaleIntensityd,
)
from monai.utils import BundleProperty, set_determinism

TEST_CASE_1 = [os.path.join(os.path.dirname(__file__), "testing_data", "inference.json")]

TEST_CASE_2 = [os.path.join(os.path.dirname(__file__), "testing_data", "inference.yaml")]


class NonConfigWorkflow(BundleWorkflow):
    """
    Test class simulates the bundle workflow defined by Python script directly.

    """

    def __init__(self, filename, output_dir):
        super().__init__(workflow="inference")
        self.filename = filename
        self.output_dir = output_dir
        self._bundle_root = "will override"
        self._device = torch.device("cpu")
        self._network_def = None
        self._inferer = None
        self._preprocessing = None
        self._postprocessing = None
        self._evaluator = None

    def initialize(self):
        set_determinism(0)
        if self._preprocessing is None:
            self._preprocessing = Compose(
                [LoadImaged(keys="image"), EnsureChannelFirstd(keys="image"), ScaleIntensityd(keys="image")]
            )
        dataset = Dataset(data=[{"image": self.filename}], transform=self._preprocessing)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

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
        if name == "device":
            return self._device
        if name == "network_def":
            return self._network_def
        if name == "inferer":
            return self._inferer
        if name == "preprocessing":
            return self._preprocessing
        if name == "postprocessing":
            return self._postprocessing
        if property[BundleProperty.REQUIRED]:
            raise ValueError(f"unsupported property '{name}' is required in the bundle properties.")

    def _set_property(self, name, property, value):
        if name == "bundle_root":
            self._bundle_root = value
        elif name == "device":
            self._device = value
        elif name == "network_def":
            self._network_def = value
        elif name == "inferer":
            self._inferer = value
        elif name == "preprocessing":
            self._preprocessing = value
        elif name == "postprocessing":
            self._postprocessing = value
        elif property[BundleProperty.REQUIRED]:
            raise ValueError(f"unsupported property '{name}' is required in the bundle properties.")


class TestBundleWorkflow(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp()
        self.expected_shape = (128, 128, 128)
        test_image = np.random.rand(*self.expected_shape)
        self.filename = os.path.join(self.data_dir, "image.nii")
        nib.save(nib.Nifti1Image(test_image, np.eye(4)), self.filename)

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def _test_inferer(self, inferer):
        # should initialize before parsing any bundle content
        inferer.initialize()
        # test required and optional properties
        inferer.check_properties()
        # test read / write the properties, note that we don't assume it as JSON or YAML config here
        self.assertEqual(inferer.bundle_root, "will override")
        self.assertEqual(inferer.device, torch.device("cpu"))
        net = inferer.network_def
        self.assertTrue(isinstance(net, UNet))
        sliding_window = inferer.inferer
        self.assertTrue(isinstance(sliding_window, SlidingWindowInferer))
        preprocessing = inferer.preprocessing
        self.assertTrue(isinstance(preprocessing, Compose))
        postprocessing = inferer.postprocessing
        self.assertTrue(isinstance(postprocessing, Compose))
        # test optional properties
        self.assertTrue(inferer.key_metric is None)
        inferer.bundle_root = "/workspace/data/spleen_ct_segmentation"
        inferer.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inferer.network_def = deepcopy(net)
        inferer.inferer = deepcopy(sliding_window)
        inferer.preprocessing = deepcopy(preprocessing)
        inferer.postprocessing = deepcopy(postprocessing)
        # test optional properties
        inferer.key_metric = "set optional properties"

        # should initialize and parse again as changed the bundle content
        inferer.initialize()
        inferer.run()
        inferer.finalize()
        # verify inference output
        loader = LoadImage(image_only=True)
        pred_file = os.path.join(self.data_dir, "image", "image_seg.nii.gz")
        self.assertTupleEqual(loader(pred_file).shape, self.expected_shape)
        os.remove(pred_file)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_config(self, config_file):
        override = {
            "network": "$@network_def.to(@device)",
            "dataset#_target_": "Dataset",
            "dataset#data": [{"image": self.filename}],
            "postprocessing#transforms#2#output_postfix": "seg",
            "output_dir": self.data_dir,
        }
        # test standard MONAI model-zoo config workflow
        inferer = ConfigWorkflow(
            workflow="infer",
            config_file=config_file,
            logging_file=os.path.join(os.path.dirname(__file__), "testing_data", "logging.conf"),
            **override,
        )
        self._test_inferer(inferer)

    def test_non_config(self):
        # test user defined python style workflow
        inferer = NonConfigWorkflow(self.filename, self.data_dir)
        self._test_inferer(inferer)


if __name__ == "__main__":
    unittest.main()

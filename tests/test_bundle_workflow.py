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

from monai.bundle import ConfigWorkflow
from monai.data import Dataset
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage
from tests.nonconfig_workflow import NonConfigWorkflow

TEST_CASE_1 = [os.path.join(os.path.dirname(__file__), "testing_data", "inference.json")]

TEST_CASE_2 = [os.path.join(os.path.dirname(__file__), "testing_data", "inference.yaml")]

TEST_CASE_3 = [os.path.join(os.path.dirname(__file__), "testing_data", "config_fl_train.json")]


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
        self.assertListEqual(inferer.check_properties(), [])
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
        # test optional properties get
        self.assertTrue(inferer.key_metric is None)
        inferer.bundle_root = "/workspace/data/spleen_ct_segmentation"
        inferer.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inferer.network_def = deepcopy(net)
        inferer.inferer = deepcopy(sliding_window)
        inferer.preprocessing = deepcopy(preprocessing)
        inferer.postprocessing = deepcopy(postprocessing)
        # test optional properties set
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
    def test_inference_config(self, config_file):
        override = {
            "network": "$@network_def.to(@device)",
            "dataset#_target_": "Dataset",
            "dataset#data": [{"image": self.filename}],
            "postprocessing#transforms#2#output_postfix": "seg",
            "output_dir": self.data_dir,
        }
        # test standard MONAI model-zoo config workflow
        inferer = ConfigWorkflow(
            workflow_type="infer",
            config_file=config_file,
            logging_file=os.path.join(os.path.dirname(__file__), "testing_data", "logging.conf"),
            **override,
        )
        self._test_inferer(inferer)

    @parameterized.expand([TEST_CASE_3])
    def test_train_config(self, config_file):
        # test standard MONAI model-zoo config workflow
        trainer = ConfigWorkflow(
            workflow_type="train",
            config_file=config_file,
            logging_file=os.path.join(os.path.dirname(__file__), "testing_data", "logging.conf"),
            init_id="initialize",
            run_id="run",
            final_id="finalize",
        )
        # should initialize before parsing any bundle content
        trainer.initialize()
        # test required and optional properties
        self.assertListEqual(trainer.check_properties(), [])
        # test override optional properties
        trainer.parser.update(
            pairs={"validate#evaluator#postprocessing": "$@validate#postprocessing if @val_interval > 0 else None"}
        )
        trainer.initialize()
        self.assertListEqual(trainer.check_properties(), [])
        # test read / write the properties
        dataset = trainer.train_dataset
        self.assertTrue(isinstance(dataset, Dataset))
        inferer = trainer.train_inferer
        self.assertTrue(isinstance(inferer, SimpleInferer))
        # test optional properties get
        self.assertTrue(trainer.train_key_metric is None)
        trainer.train_dataset = deepcopy(dataset)
        trainer.train_inferer = deepcopy(inferer)
        # test optional properties set
        trainer.train_key_metric = "set optional properties"

        # should initialize and parse again as changed the bundle content
        trainer.initialize()
        trainer.run()
        trainer.finalize()

    def test_non_config(self):
        # test user defined python style workflow
        inferer = NonConfigWorkflow(self.filename, self.data_dir)
        self._test_inferer(inferer)


if __name__ == "__main__":
    unittest.main()

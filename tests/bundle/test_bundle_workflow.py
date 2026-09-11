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

import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from parameterized import parameterized

from monai.bundle import ConfigWorkflow, create_workflow
from monai.data import Dataset
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage, LoadImaged, SaveImaged
from tests.nonconfig_workflow import NonConfigWorkflow, PythonicWorkflowImpl

MODULE_PATH = Path(__file__).resolve().parents[1]

TEST_CASE_1 = [os.path.join(MODULE_PATH, "testing_data", "inference.json")]

TEST_CASE_2 = [os.path.join(MODULE_PATH, "testing_data", "inference.yaml")]

TEST_CASE_3 = [os.path.join(MODULE_PATH, "testing_data", "config_fl_train.json")]

TEST_CASE_4 = [os.path.join(MODULE_PATH, "testing_data", "responsive_inference.json")]

TEST_CASE_NON_CONFIG_WRONG_LOG = [None, "logging.conf", "Cannot find the logging config file: logging.conf."]


class TestBundleWorkflow(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp()
        self.expected_shape = (128, 128, 128)
        test_image = np.random.rand(*self.expected_shape)
        self.filename = os.path.join(self.data_dir, "image.nii")
        self.filename1 = os.path.join(self.data_dir, "image1.nii")
        nib.save(nib.Nifti1Image(test_image, np.eye(4)), self.filename)
        nib.save(nib.Nifti1Image(test_image, np.eye(4)), self.filename1)

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
            logging_file=os.path.join(MODULE_PATH, "testing_data", "logging.conf"),
            **override,
        )
        self._test_inferer(inferer)

        # test property path
        inferer = ConfigWorkflow(
            config_file=config_file,
            workflow_type="infer",
            properties_path=os.path.join(MODULE_PATH, "testing_data", "fl_infer_properties.json"),
            logging_file=os.path.join(MODULE_PATH, "testing_data", "logging.conf"),
            **override,
        )
        self._test_inferer(inferer)
        self.assertEqual(inferer.workflow_type, "infer")

    @parameterized.expand([TEST_CASE_4])
    def test_responsive_inference_config(self, config_file):
        input_loader = LoadImaged(keys="image")
        output_saver = SaveImaged(keys="pred", output_dir=self.data_dir, output_postfix="seg")

        # test standard MONAI model-zoo config workflow
        inferer = ConfigWorkflow(
            workflow_type="infer",
            config_file=config_file,
            logging_file=os.path.join(MODULE_PATH, "testing_data", "logging.conf"),
        )
        # FIXME: temp add the property for test, we should add it to some formal realtime infer properties
        inferer.add_property(name="dataflow", required=True, config_id="dataflow")

        inferer.initialize()
        inferer.dataflow.update(input_loader({"image": self.filename}))
        inferer.run()
        output_saver(inferer.dataflow)
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "image", "image_seg.nii.gz")))

        # bundle is instantiated and idle, just change the input for next inference
        inferer.dataflow.clear()
        inferer.dataflow.update(input_loader({"image": self.filename1}))
        inferer.run()
        output_saver(inferer.dataflow)
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, "image1", "image1_seg.nii.gz")))

        inferer.finalize()

    @parameterized.expand([TEST_CASE_3])
    def test_train_config(self, config_file):
        # test standard MONAI model-zoo config workflow
        trainer = ConfigWorkflow(
            workflow_type="train",
            config_file=config_file,
            logging_file=os.path.join(MODULE_PATH, "testing_data", "logging.conf"),
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
        self.assertIsInstance(dataset, Dataset)
        inferer = trainer.train_inferer
        self.assertIsInstance(inferer, SimpleInferer)
        # test optional properties get
        self.assertIsNone(trainer.train_key_metric)
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
        self.assertEqual(inferer.meta_file, None)
        self._test_inferer(inferer)

    @parameterized.expand([TEST_CASE_NON_CONFIG_WRONG_LOG])
    def test_non_config_wrong_log_cases(self, meta_file, logging_file, expected_error):
        with self.assertRaisesRegex(FileNotFoundError, expected_error):
            NonConfigWorkflow(self.filename, self.data_dir, meta_file, logging_file)

    def test_pythonic_workflow(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_file = {"roi_size": (64, 64, 32)}
        meta_file = os.path.join(MODULE_PATH, "testing_data", "metadata.json")
        property_path = os.path.join(MODULE_PATH, "testing_data", "python_workflow_properties.json")
        workflow = PythonicWorkflowImpl(
            workflow_type="infer", config_file=config_file, meta_file=meta_file, properties_path=property_path
        )
        workflow.initialize()
        # Load input data
        input_loader = LoadImaged(keys="image")
        workflow.dataflow.update(input_loader({"image": self.filename}))
        self.assertEqual(workflow.bundle_root, ".")
        self.assertEqual(workflow.device, device)
        self.assertEqual(workflow.version, "0.1.0")
        # check config override correctly
        self.assertEqual(workflow.inferer.roi_size, (64, 64, 32))
        workflow.run()
        # update input data and run again
        workflow.dataflow.update(input_loader({"image": self.filename1}))
        workflow.run()
        pred = workflow.dataflow["pred"]
        self.assertEqual(pred.shape[2:], self.expected_shape)
        self.assertEqual(pred.meta["filename_or_obj"], self.filename1)
        workflow.finalize()

    def test_create_pythonic_workflow(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_file = {"roi_size": (64, 64, 32)}
        meta_file = os.path.join(MODULE_PATH, "testing_data", "metadata.json")
        property_path = os.path.join(MODULE_PATH, "testing_data", "python_workflow_properties.json")
        sys.path.append(MODULE_PATH)
        workflow = create_workflow(
            "tests.nonconfig_workflow.PythonicWorkflowImpl",
            workflow_type="infer",
            config_file=config_file,
            meta_file=meta_file,
            properties_path=property_path,
        )
        # Load input data
        input_loader = LoadImaged(keys="image")
        workflow.dataflow.update(input_loader({"image": self.filename}))
        self.assertEqual(workflow.bundle_root, ".")
        self.assertEqual(workflow.device, device)
        self.assertEqual(workflow.version, "0.1.0")
        # check config override correctly
        self.assertEqual(workflow.inferer.roi_size, (64, 64, 32))

        # check set property override correctly
        workflow.inferer = SlidingWindowInferer(roi_size=config_file["roi_size"], sw_batch_size=1, overlap=0.5)
        workflow.initialize()
        self.assertEqual(workflow.inferer.overlap, 0.5)

        workflow.run()
        # update input data and run again
        workflow.dataflow.update(input_loader({"image": self.filename1}))
        workflow.run()
        pred = workflow.dataflow["pred"]
        self.assertEqual(pred.shape[2:], self.expected_shape)
        self.assertEqual(pred.meta["filename_or_obj"], self.filename1)

        # test add properties
        workflow.add_property(name="net", required=True, desc="network for the training.")
        self.assertIn("net", workflow.properties)
        workflow.finalize()


class TestConfigWorkflowWarnsOnLoggingConf(unittest.TestCase):
    """Regression test for GHSA-wvpx-5qmp-46g3: `ConfigWorkflow` defaults `logging_file` to the
    bundle's own "configs/logging.conf" and hands it to `logging.config.fileConfig`, which `eval()`s
    the INI's `class=`/`args=` fields. It fires in `__init__`, before `initialize()` or `run()`, and
    lives in a plain INI rather than the MONAI `$`-DSL, so it is easy to miss when reviewing a
    bundle. `class=` is now restricted to the stdlib logging namespaces and `args=`/`kwargs=` to
    literals plus `sys.stdout`/`sys.stderr`, so a config that would execute code is rejected before
    `fileConfig` ever sees it."""

    def setUp(self):
        # `fileConfig` reconfigures logging process-wide. Snapshot the root logger and restore it
        # afterwards so these tests cannot leak a handler into the rest of the suite.
        root = logging.getLogger()
        level, handlers, filters = root.level, root.handlers[:], root.filters[:]
        disabled = logging.root.manager.disable

        def _restore():
            # Detach whatever is on the root logger now, closing anything `fileConfig` installed so
            # it does not linger in logging's handler registry, then put the snapshot back. Under
            # `tests/runner.py` the root logger starts with no handlers, so there is nothing for
            # `fileConfig` to have closed on the way in.
            for handler in root.handlers[:]:
                root.removeHandler(handler)
                if handler not in handlers:
                    handler.close()
            root.setLevel(level)
            root.filters[:] = filters
            for handler in handlers:
                root.addHandler(handler)
            logging.disable(disabled)

        self.addCleanup(_restore)

    def test_class_call_expr_is_rejected(self):
        """A ``class=`` call expression (no dot, no tuple subscript) is refused.

        This is the exact shape the previous string-prefix check could not see: ``rsplit('.', 1)[0]``
        on a call with no period returns the whole expression, and `class=` values in real configs
        are dotted logging names, so such a call would never have been allowed -- it is rejected
        here by the AST parser rather than the prefix test.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            configs = os.path.join(tempdir, "configs")
            os.makedirs(configs)
            marker = os.path.join(tempdir, "PWNED")
            with open(os.path.join(configs, "train.json"), "w") as f:
                json.dump({"initialize": []}, f)
            payload = f"__import__('pathlib').Path({marker!r}).write_text('pwned')"
            with open(os.path.join(configs, "logging.conf"), "w") as f:
                f.write(
                    "[loggers]\nkeys=root\n[handlers]\nkeys=h\n[formatters]\nkeys=f\n"
                    "[logger_root]\nlevel=NOTSET\nhandlers=h\n"
                    f"[handler_h]\nclass={payload}()\nargs=()\nformatter=f\n"
                    "[formatter_f]\nformat=%(message)s\n"
                )
            with self.assertRaisesRegex(ValueError, r"GHSA-wvpx-5qmp-46g3"):
                ConfigWorkflow(config_file=os.path.join(configs, "train.json"), workflow_type="train")
            self.assertFalse(os.path.exists(marker), "the logging.conf payload executed")

    def test_class_attribute_call_is_rejected(self):
        """A ``class=`` attribute-chain expression is refused even when it is not a bare call.

        The root module is not on the `logging` allowlist, so the attribute chain is rejected by
        the AST check regardless of the trailing call.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            configs = os.path.join(tempdir, "configs")
            os.makedirs(configs)
            marker = os.path.join(tempdir, "PWNED")
            with open(os.path.join(configs, "train.json"), "w") as f:
                json.dump({"initialize": []}, f)
            payload = f"__import__('pathlib').Path({marker!r}).write_text"
            with open(os.path.join(configs, "logging.conf"), "w") as f:
                f.write(
                    "[loggers]\nkeys=root\n[handlers]\nkeys=h\n[formatters]\nkeys=f\n"
                    "[logger_root]\nlevel=NOTSET\nhandlers=h\n"
                    f"[handler_h]\nclass={payload}\nargs=()\nformatter=f\n"
                    "[formatter_f]\nformat=%(message)s\n"
                )
            with self.assertRaisesRegex(ValueError, r"GHSA-wvpx-5qmp-46g3"):
                ConfigWorkflow(config_file=os.path.join(configs, "train.json"), workflow_type="train")
            self.assertFalse(os.path.exists(marker), "the logging.conf payload executed")

    def test_bare_eval_class_is_rejected(self):
        """A bare ``class=eval`` is refused before it can run an ``args=`` payload.

        `fileConfig` resolves a dotless ``class=`` against the `logging` module's namespace via
        ``eval()``, so ``class=eval`` paired with a literal ``args=`` tuple -- which the literal
        check permits, because the tuple itself is inert -- would hand an attacker-controlled
        string straight to ``eval``. Accepting any bare name is therefore not safe: the name must
        resolve to a real handler or formatter class.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            configs = os.path.join(tempdir, "configs")
            os.makedirs(configs)
            marker = os.path.join(tempdir, "PWNED")
            with open(os.path.join(configs, "train.json"), "w") as f:
                json.dump({"initialize": []}, f)
            payload = f"__import__('pathlib').Path({marker!r}).write_text('pwned')"
            with open(os.path.join(configs, "logging.conf"), "w") as f:
                f.write(
                    "[loggers]\nkeys=root\n[handlers]\nkeys=h\n[formatters]\nkeys=f\n"
                    "[logger_root]\nlevel=NOTSET\nhandlers=h\n"
                    f"[handler_h]\nclass=eval\nargs=({payload!r},)\nformatter=f\n"
                    "[formatter_f]\nformat=%(message)s\n"
                )
            with self.assertRaisesRegex(ValueError, r"GHSA-wvpx-5qmp-46g3"):
                ConfigWorkflow(config_file=os.path.join(configs, "train.json"), workflow_type="train")
            self.assertFalse(os.path.exists(marker), "the logging.conf payload executed")

    def test_non_handler_logging_attribute_is_rejected(self):
        """A real `logging` attribute that is not a handler or formatter is refused.

        ``logging.Logger`` lives in an allowlisted module but is not a `Handler`/`Formatter`
        subclass, so allowing the module alone is too coarse a boundary.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            configs = os.path.join(tempdir, "configs")
            os.makedirs(configs)
            with open(os.path.join(configs, "train.json"), "w") as f:
                json.dump({"initialize": []}, f)
            with open(os.path.join(configs, "logging.conf"), "w") as f:
                f.write(
                    "[loggers]\nkeys=root\n[handlers]\nkeys=h\n[formatters]\nkeys=f\n"
                    "[logger_root]\nlevel=NOTSET\nhandlers=h\n"
                    "[handler_h]\nclass=logging.Logger\nargs=()\nformatter=f\n"
                    "[formatter_f]\nformat=%(message)s\n"
                )
            with self.assertRaisesRegex(ValueError, r"GHSA-wvpx-5qmp-46g3"):
                ConfigWorkflow(config_file=os.path.join(configs, "train.json"), workflow_type="train")

    def test_rotating_file_handler_is_accepted(self):
        """A dotted `logging.handlers` handler keeps working under the class allowlist."""
        with tempfile.TemporaryDirectory() as tempdir:
            configs = os.path.join(tempdir, "configs")
            os.makedirs(configs)
            logfile = os.path.join(tempdir, "run.log")
            with open(os.path.join(configs, "train.json"), "w") as f:
                json.dump({"initialize": []}, f)
            with open(os.path.join(configs, "logging.conf"), "w") as f:
                f.write(
                    "[loggers]\nkeys=root\n[handlers]\nkeys=h\n[formatters]\nkeys=f\n"
                    "[logger_root]\nlevel=INFO\nhandlers=h\n"
                    "[handler_h]\nclass=logging.handlers.RotatingFileHandler\n"
                    f"level=INFO\nformatter=f\nargs=({logfile!r},)\n"
                    "[formatter_f]\nformat=%(message)s\n"
                )
            ConfigWorkflow(config_file=os.path.join(configs, "train.json"), workflow_type="train")

    def test_default_logging_conf_payload_is_rejected(self):
        """The `class=` payload is refused and never runs."""
        with tempfile.TemporaryDirectory() as tempdir:
            configs = os.path.join(tempdir, "configs")
            os.makedirs(configs)
            marker = os.path.join(tempdir, "PWNED")
            with open(os.path.join(configs, "train.json"), "w") as f:
                json.dump({"initialize": []}, f)
            # `fileConfig` would eval() the `class=` field, so the tuple subscript runs the payload
            # and still yields a usable handler class -- unless the config is rejected first.
            with open(os.path.join(configs, "logging.conf"), "w") as f:
                f.write(
                    "[loggers]\nkeys=root\n[handlers]\nkeys=h\n[formatters]\nkeys=f\n"
                    "[logger_root]\nlevel=NOTSET\nhandlers=h\n"
                    "[handler_h]\n"
                    f"class=(__import__('pathlib').Path({marker!r}).write_text('pwned'), "
                    "__import__('logging').StreamHandler)[1]\nargs=()\nformatter=f\n"
                    "[formatter_f]\nformat=%(message)s\n"
                )
            with self.assertRaisesRegex(ValueError, r"GHSA-wvpx-5qmp-46g3"):
                ConfigWorkflow(config_file=os.path.join(configs, "train.json"), workflow_type="train")
            self.assertFalse(os.path.exists(marker), "the logging.conf payload executed")

    def test_args_payload_is_rejected(self):
        """A payload hidden in `args=` is refused even with an allowlisted `class=`."""
        with tempfile.TemporaryDirectory() as tempdir:
            configs = os.path.join(tempdir, "configs")
            os.makedirs(configs)
            marker = os.path.join(tempdir, "PWNED")
            with open(os.path.join(configs, "train.json"), "w") as f:
                json.dump({"initialize": []}, f)
            with open(os.path.join(configs, "logging.conf"), "w") as f:
                f.write(
                    "[loggers]\nkeys=root\n[handlers]\nkeys=h\n[formatters]\nkeys=f\n"
                    "[logger_root]\nlevel=NOTSET\nhandlers=h\n"
                    "[handler_h]\nclass=StreamHandler\n"
                    f"args=(__import__('pathlib').Path({marker!r}).write_text('pwned'),)\nformatter=f\n"
                    "[formatter_f]\nformat=%(message)s\n"
                )
            with self.assertRaisesRegex(ValueError, r"GHSA-wvpx-5qmp-46g3"):
                ConfigWorkflow(config_file=os.path.join(configs, "train.json"), workflow_type="train")
            self.assertFalse(os.path.exists(marker), "the logging.conf payload executed")

    def test_benign_logging_conf_still_applies(self):
        """The standard `class=StreamHandler` / `args=(sys.stdout,)` form keeps working."""
        with tempfile.TemporaryDirectory() as tempdir:
            configs = os.path.join(tempdir, "configs")
            os.makedirs(configs)
            with open(os.path.join(configs, "train.json"), "w") as f:
                json.dump({"initialize": []}, f)
            with open(os.path.join(configs, "logging.conf"), "w") as f:
                f.write(
                    "[loggers]\nkeys=root\n[handlers]\nkeys=h\n[formatters]\nkeys=f\n"
                    "[logger_root]\nlevel=INFO\nhandlers=h\n"
                    "[handler_h]\nclass=StreamHandler\nlevel=INFO\nformatter=f\nargs=(sys.stdout,)\n"
                    "[formatter_f]\nformat=%(asctime)s - %(message)s\n"
                )
            # Must not raise.
            ConfigWorkflow(config_file=os.path.join(configs, "train.json"), workflow_type="train")

    def test_no_warning_when_logging_disabled(self):
        """No warning when `fileConfig` is never reached -- the file exists but is opted out of."""
        with tempfile.TemporaryDirectory() as tempdir:
            configs = os.path.join(tempdir, "configs")
            os.makedirs(configs)
            marker = os.path.join(tempdir, "PWNED")
            with open(os.path.join(configs, "train.json"), "w") as f:
                json.dump({"initialize": []}, f)
            with open(os.path.join(configs, "logging.conf"), "w") as f:
                f.write(
                    "[loggers]\nkeys=root\n[handlers]\nkeys=h\n[formatters]\nkeys=f\n"
                    "[logger_root]\nlevel=NOTSET\nhandlers=h\n"
                    "[handler_h]\n"
                    f"class=(__import__('pathlib').Path({marker!r}).write_text('pwned'), "
                    "__import__('logging').StreamHandler)[1]\nargs=()\nformatter=f\n"
                    "[formatter_f]\nformat=%(message)s\n"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                ConfigWorkflow(
                    config_file=os.path.join(configs, "train.json"), workflow_type="train", logging_file=False
                )
            self.assertFalse(os.path.exists(marker))


if __name__ == "__main__":
    unittest.main()

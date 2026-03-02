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

import unittest

import torch
from parameterized import parameterized

from monai.handlers import CalibrationError, from_engine
from monai.utils import IgniteInfo, min_version, optional_import
from tests.test_utils import assert_allclose

Engine, has_ignite = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")

_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Test cases for handler
# Format: [input_params, expected_value, expected_rows, expected_channels]
TEST_CASE_1 = [
    {
        "num_bins": 5,
        "include_background": True,
        "calibration_reduction": "expected",
        "metric_reduction": "mean",
        "output_transform": from_engine(["pred", "label"]),
    },
    0.2250,
    4,  # 2 batches * 2 iterations
    2,  # 2 channels
]

TEST_CASE_2 = [
    {
        "num_bins": 5,
        "include_background": False,
        "calibration_reduction": "expected",
        "metric_reduction": "mean",
        "output_transform": from_engine(["pred", "label"]),
    },
    0.2500,
    4,  # 2 batches * 2 iterations
    1,  # 1 channel (background excluded)
]

TEST_CASE_3 = [
    {
        "num_bins": 5,
        "include_background": True,
        "calibration_reduction": "average",
        "metric_reduction": "mean",
        "output_transform": from_engine(["pred", "label"]),
    },
    0.2584,  # Mean of [[0.2000, 0.4667], [0.2000, 0.1667]]
    4,
    2,
]

TEST_CASE_4 = [
    {
        "num_bins": 5,
        "include_background": True,
        "calibration_reduction": "maximum",
        "metric_reduction": "mean",
        "output_transform": from_engine(["pred", "label"]),
    },
    0.4000,  # Mean of [[0.3000, 0.7000], [0.3000, 0.3000]]
    4,
    2,
]


@unittest.skipUnless(has_ignite, "Requires pytorch-ignite")
class TestHandlerCalibrationError(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_compute(self, input_params, expected_value, expected_rows, expected_channels):
        calibration_metric = CalibrationError(**input_params)

        # Test data: 2 batches with 2 channels each
        y_pred = torch.tensor(
            [[[[0.7, 0.3], [0.1, 0.9]], [[0.7, 0.3], [0.5, 0.5]]], [[[0.9, 0.9], [0.3, 0.3]], [[0.1, 0.1], [0.9, 0.7]]]]
        ).to(_device)
        y = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]], [[[1, 1], [0, 0]], [[0, 0], [1, 1]]]]).to(_device)

        # Create data as list of batches (2 iterations)
        data = [{"pred": y_pred, "label": y}, {"pred": y_pred, "label": y}]

        def _val_func(engine, batch):
            return batch

        engine = Engine(_val_func)
        calibration_metric.attach(engine=engine, name="calibration_error")

        engine.run(data, max_epochs=1)

        assert_allclose(
            engine.state.metrics["calibration_error"], expected_value, atol=1e-4, rtol=1e-4, type_test=False
        )

        # Check details shape using invariants rather than exact tuple
        details = engine.state.metric_details["calibration_error"]
        self.assertEqual(details.shape[0], expected_rows)
        self.assertEqual(details.shape[-1], expected_channels)


@unittest.skipUnless(has_ignite, "Requires pytorch-ignite")
class TestHandlerCalibrationErrorEdgeCases(unittest.TestCase):

    def test_single_iteration(self):
        """Test handler with single iteration."""
        calibration_metric = CalibrationError(
            num_bins=5,
            include_background=True,
            calibration_reduction="expected",
            metric_reduction="mean",
            output_transform=from_engine(["pred", "label"]),
        )

        y_pred = torch.tensor([[[[0.7, 0.3], [0.1, 0.9]]]]).to(_device)
        y = torch.tensor([[[[1, 0], [0, 1]]]]).to(_device)

        data = [{"pred": y_pred, "label": y}]

        def _val_func(engine, batch):
            return batch

        engine = Engine(_val_func)
        calibration_metric.attach(engine=engine, name="calibration_error")

        engine.run(data, max_epochs=1)

        assert_allclose(engine.state.metrics["calibration_error"], 0.2, atol=1e-4, rtol=1e-4, type_test=False)

    def test_save_details_false(self):
        """Test handler with save_details=False."""
        calibration_metric = CalibrationError(
            num_bins=5,
            include_background=True,
            calibration_reduction="expected",
            metric_reduction="mean",
            output_transform=from_engine(["pred", "label"]),
            save_details=False,
        )

        y_pred = torch.tensor([[[[0.7, 0.3], [0.1, 0.9]]]]).to(_device)
        y = torch.tensor([[[[1, 0], [0, 1]]]]).to(_device)

        data = [{"pred": y_pred, "label": y}]

        def _val_func(engine, batch):
            return batch

        engine = Engine(_val_func)
        calibration_metric.attach(engine=engine, name="calibration_error")

        engine.run(data, max_epochs=1)

        assert_allclose(engine.state.metrics["calibration_error"], 0.2, atol=1e-4, rtol=1e-4, type_test=False)

        # When save_details=False, metric_details should not exist or should not have the metric key
        if hasattr(engine.state, "metric_details"):
            self.assertNotIn("calibration_error", engine.state.metric_details or {})


if __name__ == "__main__":
    unittest.main()

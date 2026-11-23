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

import csv
import os
import tempfile
import unittest
from pathlib import Path

import torch

from monai.handlers.utils import write_metrics_reports


class TestWriteMetricsReports(unittest.TestCase):

    def test_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            write_metrics_reports(
                save_dir=Path(tempdir),
                images=["filepath1", "filepath2"],
                metrics={"metric1": 1, "metric2": 2},
                metric_details={"metric3": torch.tensor([[1, 2], [2, 3]]), "metric4": torch.tensor([[5, 6], [7, 8]])},
                summary_ops=["mean", "median", "max", "90percentile"],
                deli="\t",
                output_type="csv",
            )

            # check the metrics.csv and content
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metrics.csv")))
            with open(os.path.join(tempdir, "metrics.csv")) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    self.assertEqual(row, [f"metric{i + 1}\t{i + 1}"])
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric3_raw.csv")))
            # check the metric_raw.csv and content
            with open(os.path.join(tempdir, "metric3_raw.csv")) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    if i > 0:
                        self.assertEqual(row, [f"filepath{i}\t{float(i):.4f}\t{float(i + 1):.4f}\t{i + 0.5:.4f}"])
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric3_summary.csv")))
            # check the metric_summary.csv and content
            with open(os.path.join(tempdir, "metric3_summary.csv")) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    if i == 1:
                        self.assertEqual(row, ["class0\t1.5000\t1.5000\t2.0000\t1.9000"])
                    elif i == 2:
                        self.assertEqual(row, ["class1\t2.5000\t2.5000\t3.0000\t2.9000"])
                    elif i == 3:
                        self.assertEqual(row, ["mean\t2.0000\t2.0000\t2.5000\t2.4000"])
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric4_raw.csv")))
            self.assertTrue(os.path.exists(os.path.join(tempdir, "metric4_summary.csv")))

    def test_multi_metric_details_headers(self):
        with tempfile.TemporaryDirectory() as tempdir:
            write_metrics_reports(
                save_dir=Path(tempdir),
                images=["img1", "img2"],
                metrics=None,
                metric_details={
                    "m1": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                    "m2": torch.tensor([[7, 8], [9, 10]]),
                    "m3": torch.tensor([[11, 12, 13, 14], [15, 16, 17, 18]]),
                },
                summary_ops=None,
                deli=",",
                output_type="csv",
            )
            for name, nclass in [("m1", 3), ("m2", 2), ("m3", 4)]:
                path = os.path.join(tempdir, f"{name}_raw.csv")
                self.assertTrue(os.path.exists(path))
                with open(path) as f:
                    header = f.readline().strip().split(",")
                self.assertEqual(header, ["filename"] + [f"class{i}" for i in range(nclass)] + ["mean"])

    def test_utf8_non_ascii_reports(self):
        patient_names = ["\u60a3\u8005_01_\U0001f600.nii.gz", "\u60a3\u8005_02_\u75c5\u7076.nii.gz"]
        class_labels = ["\u6b63\u5e38", "\u75c5\u7076\U0001f600"]
        with tempfile.TemporaryDirectory() as tempdir:
            save_dir = Path(tempdir) / "\u8bc4\u6d4b\u62a5\u544a"
            write_metrics_reports(
                save_dir=save_dir,
                images=patient_names,
                metrics={"dice": torch.tensor(0.75)},
                metric_details={"dice": torch.tensor([[0.5, 0.25], [0.75, 0.8]])},
                summary_ops=["mean"],
                class_labels=class_labels,
            )

            metrics_text = (save_dir / "metrics.csv").read_text(encoding="utf-8")
            raw_text = (save_dir / "dice_raw.csv").read_text(encoding="utf-8")
            summary_text = (save_dir / "dice_summary.csv").read_text(encoding="utf-8")

        self.assertTrue(metrics_text.startswith("dice,"))
        self.assertIn(patient_names[0], raw_text)
        self.assertIn(",".join(class_labels), raw_text)
        self.assertIn(class_labels[0], summary_text)
        self.assertIn(class_labels[1], summary_text)

    def test_utf8_metric_name_in_output_filename(self):
        metric_name = "dice_\u75c5\u7076"
        patient_name = "\u60a3\u8005_03_\u75c5\u7076.nii.gz"
        with tempfile.TemporaryDirectory() as tempdir:
            save_dir = Path(tempdir)
            write_metrics_reports(
                save_dir=save_dir,
                images=[patient_name],
                metrics=None,
                metric_details={metric_name: torch.tensor([[0.75]])},
                summary_ops=None,
            )

            raw_path = save_dir / f"{metric_name}_raw.csv"
            raw_text = raw_path.read_text(encoding="utf-8")

        self.assertTrue(raw_path.name.startswith(metric_name))
        self.assertIn(patient_name, raw_text)


if __name__ == "__main__":
    unittest.main()

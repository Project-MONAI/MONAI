# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
import os
import random
import tempfile
import unittest

import torch
import torch.distributed as dist
from ignite.engine import Engine, Events

from monai.handlers import MetricsSaver
from tests.utils import DistCall, DistTestCase, SkipIfBeforePyTorchVersion


@SkipIfBeforePyTorchVersion((1, 7))
class DistributedMetricsSaver(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_content(self):
        self._run()

    def _run(self):
        with tempfile.TemporaryDirectory() as tempdir:
            metrics_saver = MetricsSaver(
                save_dir=tempdir,
                metrics=["metric1", "metric2"],
                metric_details=["metric3", "metric4"],
                batch_transform=lambda x: x["image_meta_dict"],
                summary_ops="*",
            )

            def _val_func(engine, batch):
                pass

            engine = Engine(_val_func)

            # test the case that all_gather with string length > 1024 chars
            filename_postfix = "abcdefghigklmnopqrstuvwxyz"
            for _ in range(1100):
                filename_postfix += filename_postfix[random.randint(0, 26)]

            if dist.get_rank() == 0:
                data = [{"image_meta_dict": {"filename_or_obj": [f"1{filename_postfix}"]}}]

                @engine.on(Events.EPOCH_COMPLETED)
                def _save_metrics0(engine):
                    engine.state.metrics = {"metric1": 1, "metric2": 2}
                    engine.state.metric_details = {
                        "metric3": torch.tensor([[1, 2]]),
                        "metric4": torch.tensor([[5, 6]]),
                    }

            if dist.get_rank() == 1:
                # different ranks have different data length
                data = [
                    {"image_meta_dict": {"filename_or_obj": [f"2{filename_postfix}"]}},
                    {"image_meta_dict": {"filename_or_obj": [f"3{filename_postfix}"]}},
                ]

                @engine.on(Events.EPOCH_COMPLETED)
                def _save_metrics1(engine):
                    engine.state.metrics = {"metric1": 1, "metric2": 2}
                    engine.state.metric_details = {
                        "metric3": torch.tensor([[2, 3], [3, 4]]),
                        "metric4": torch.tensor([[6, 7], [7, 8]]),
                    }

            metrics_saver.attach(engine)
            engine.run(data, max_epochs=1)

            if dist.get_rank() == 0:
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
                            expected = [f"{i}{filename_postfix[0: 1023]}\t{float(i)}\t{float(i + 1)}\t{i + 0.5}"]
                            self.assertEqual(row, expected)
                self.assertTrue(os.path.exists(os.path.join(tempdir, "metric3_summary.csv")))
                # check the metric_summary.csv and content
                with open(os.path.join(tempdir, "metric3_summary.csv")) as f:
                    f_csv = csv.reader(f)
                    for i, row in enumerate(f_csv):
                        if i == 1:
                            self.assertEqual(row, ["class0\t1.0000\t1.0000\t1.0000\t1.0000\t1.0000\t0.0000"])
                        elif i == 2:
                            self.assertEqual(row, ["class1\t2.0000\t2.0000\t2.0000\t2.0000\t2.0000\t0.0000"])
                        elif i == 3:
                            self.assertEqual(row, ["mean\t1.5000\t1.5000\t1.5000\t1.5000\t1.5000\t0.0000"])
                self.assertTrue(os.path.exists(os.path.join(tempdir, "metric4_raw.csv")))
                self.assertTrue(os.path.exists(os.path.join(tempdir, "metric4_summary.csv")))


if __name__ == "__main__":
    unittest.main()

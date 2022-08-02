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

import datetime
import os
import unittest

import torch

from tests.utils import SkipIfNoModule

import monai.transforms as mt
from monai.engines import SupervisedTrainer
from monai.utils import optional_import
from monai.utils.enums import CommonKeys
from monai.utils.profiling import ProfileResult, WorkflowProfiler, ProfileHandler
from io import StringIO

pd, _ = optional_import("pandas")


class TestWorkflowProfiler(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.scale = mt.ScaleIntensity()
        self.test_comp = mt.Compose([mt.ScaleIntensity(), mt.RandAxisFlip(0.5)])
        self.test_image = torch.rand(1, 16, 16, 16)
        self.pid = os.getpid()

    def test_empty(self):
        wp = WorkflowProfiler()

        with wp:
            pass

        self.assertEqual(wp.get_results(), {})

    def test_profile_transforms(self):
        call_name = "ScaleIntensity.__call__"
        with WorkflowProfiler() as wp:
            self.scale(self.test_image)

        results = wp.get_results()
        self.assertSequenceEqual(list(results), [call_name])

        prs = results[call_name]

        self.assertEqual(len(prs), 1)

        pr = prs[0]

        self.assertIsInstance(pr, ProfileResult)
        self.assertEqual(pr.name, call_name)
        self.assertEqual(pr.pid, self.pid)
        self.assertGreater(pr.time, 0)

        dt = datetime.datetime.fromisoformat(pr.timestamp)

        self.assertIsInstance(dt, datetime.datetime)

    def test_profile_context(self):
        with WorkflowProfiler() as wp:
            with wp.profile_ctx("context"):
                self.scale(self.test_image)

            with wp.profile_ctx("context"):
                self.scale(self.test_image)

        results = wp.get_results()
        self.assertSequenceEqual(set(results), {"ScaleIntensity.__call__", "context"})

        prs = results["context"]

        self.assertEqual(len(prs), 2)

    def test_profile_callable(self):
        def funca():
            pass

        with WorkflowProfiler() as wp:
            funca = wp.profile_callable()(funca)

            funca()

            @wp.profile_callable("funcb")
            def _func():
                pass

            _func()
            _func()

        results = wp.get_results()
        self.assertSequenceEqual(set(results), {"funca", "funcb"})

        self.assertEqual(len(results["funca"]), 1)
        self.assertEqual(len(results["funcb"]), 2)

    def test_profile_iteration(self):
        with WorkflowProfiler() as wp:
            range_vals = []

            for i in wp.profile_iter("range5", range(5)):
                range_vals.append(i)

            self.assertSequenceEqual(range_vals, list(range(5)))

        results = wp.get_results()
        self.assertSequenceEqual(set(results), {"range5"})

        self.assertEqual(len(results["range5"]), 5)

    def test_times_summary(self):
        call_name = "ScaleIntensity.__call__"

        with WorkflowProfiler() as wp:
            self.scale(self.test_image)

        tsum = wp.get_times_summary()

        self.assertSequenceEqual(list(tsum), [call_name])

        times = tsum[call_name]

        self.assertEqual(len(times), 6)
        self.assertEqual(times[0], 1)

    @SkipIfNoModule("pandas")
    def test_times_summary_pd(self):
        with WorkflowProfiler() as wp:
            self.scale(self.test_image)

        df = wp.get_times_summary_pd()

        self.assertIsInstance(df, pd.DataFrame)

    def test_csv_dump(self):
        with WorkflowProfiler() as wp:
            self.scale(self.test_image)
            
        sio=StringIO()
        wp.dump_csv(sio)
        self.assertGreater(sio.tell(),0)
    
    @SkipIfNoModule("ignite")
    def test_handler(self):
        from ignite.engine import Events
        
        net=torch.nn.Conv2d(1,1,3,padding=1)
        im=torch.rand(1,1,16,16)
        
        with WorkflowProfiler(None) as wp:
            trainer=SupervisedTrainer(
                device=torch.device("cpu"),
                max_epochs=2,
                train_data_loader=[{CommonKeys.IMAGE:im,CommonKeys.LABEL:im}]*3,
                epoch_length=3,
                network=net,
                optimizer=torch.optim.Adam(net.parameters()),
                loss_function=torch.nn.L1Loss(),
            )
            
            epoch_h=ProfileHandler("Epoch",wp, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED).attach(trainer)

            trainer.run()
            
        results = wp.get_results()
        
        self.assertSequenceEqual(set(results), {"Epoch"})
        self.assertEqual(len(results["Epoch"]), 2)
        
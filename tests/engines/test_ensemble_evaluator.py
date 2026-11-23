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

import threading
import unittest

import torch
from ignite.engine import EventEnum, Events
from parameterized import parameterized

from monai.engines import EnsembleEvaluator
from tests.test_utils import assert_allclose

TEST_CASE_1 = [["pred_0", "pred_1", "pred_2", "pred_3", "pred_4"]]

TEST_CASE_2 = [None]


class TestEnsembleEvaluator(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_content(self, pred_keys):
        device = torch.device("cpu:0")

        class TestDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, index):
                return {"image": torch.tensor([index]), "label": torch.zeros(1)}

        val_loader = torch.utils.data.DataLoader(TestDataset())

        class TestNet(torch.nn.Module):
            def __init__(self, func):
                super().__init__()
                self.func = func

            def forward(self, x):
                return self.func(x)

        net0 = TestNet(lambda x: x + 1)
        net1 = TestNet(lambda x: x + 2)
        net2 = TestNet(lambda x: x + 3)
        net3 = TestNet(lambda x: x + 4)
        net4 = TestNet(lambda x: x + 5)

        class CustomEvents(EventEnum):
            FOO_EVENT = "foo_event"
            BAR_EVENT = "bar_event"

        val_engine = EnsembleEvaluator(
            device=device,
            val_data_loader=val_loader,
            networks=[net0, net1, net2, net3, net4],
            pred_keys=pred_keys,
            event_names=["bwd_event", "opt_event", CustomEvents],
            event_to_attr={CustomEvents.FOO_EVENT: "foo", "opt_event": "opt"},
        )

        @val_engine.on(Events.ITERATION_COMPLETED)
        def run_transform(engine):
            for i in range(5):
                expected_value = engine.state.iteration + i
                assert_allclose(engine.state.output[0][f"pred_{i}"].item(), expected_value)

        @val_engine.on(Events.EPOCH_COMPLETED)
        def trigger_custom_event():
            val_engine.fire_event(CustomEvents.FOO_EVENT)
            val_engine.fire_event(CustomEvents.BAR_EVENT)
            val_engine.fire_event("bwd_event")
            val_engine.fire_event("opt_event")

        @val_engine.on(CustomEvents.FOO_EVENT)
        def do_foo_op():
            self.assertEqual(val_engine.state.foo, 0)

        @val_engine.on("opt_event")
        def do_bar_op():
            self.assertEqual(val_engine.state.opt, 0)

        val_engine.run()


class TestEnsembleEvaluatorParallel(unittest.TestCase):
    def _get_loader(self):
        class TestDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return {"image": torch.tensor([index]), "label": torch.zeros(1)}

        return torch.utils.data.DataLoader(TestDataset())

    def test_parallel_inference_and_output_order(self):
        barrier = threading.Barrier(2, timeout=5)
        thread_ids = set()
        thread_ids_lock = threading.Lock()

        class BarrierNet(torch.nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset

            def forward(self, x):
                with thread_ids_lock:
                    thread_ids.add(threading.get_ident())
                barrier.wait()
                return x + self.offset

        val_engine = EnsembleEvaluator(
            device=torch.device("cpu:0"),
            val_data_loader=self._get_loader(),
            networks=[BarrierNet(1), BarrierNet(2)],
            parallel=True,
            max_workers=2,
        )

        @val_engine.on(Events.ITERATION_COMPLETED)
        def check_output(engine):
            expected_batch = torch.tensor([engine.state.iteration - 1], dtype=torch.float32)
            assert_allclose(engine.state.output["pred_0"], expected_batch + 1)
            assert_allclose(engine.state.output["pred_1"], expected_batch + 2)

        val_engine.run()
        self.assertEqual(len(thread_ids), 2)

    def test_parallel_run_can_be_reused(self):
        class AddNet(torch.nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset

            def forward(self, x):
                return x + self.offset

        val_engine = EnsembleEvaluator(
            device=torch.device("cpu:0"),
            val_data_loader=self._get_loader(),
            networks=[AddNet(1), AddNet(2)],
            parallel=True,
        )
        iterations = []

        @val_engine.on(Events.ITERATION_COMPLETED)
        def record_iteration(engine):
            iterations.append(engine.state.output["pred_0"].item())

        val_engine.run()
        val_engine.run()
        self.assertEqual(iterations, [1, 2, 3, 4, 1, 2, 3, 4])

    def test_parallel_inference_propagates_errors(self):
        class FailingNet(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("inference failed")

        val_engine = EnsembleEvaluator(
            device=torch.device("cpu:0"),
            val_data_loader=self._get_loader(),
            networks=[torch.nn.Identity(), FailingNet()],
            parallel=True,
        )
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            val_engine.run()

    def test_invalid_max_workers(self):
        with self.assertRaisesRegex(ValueError, "max_workers"):
            EnsembleEvaluator(
                device=torch.device("cpu:0"),
                val_data_loader=self._get_loader(),
                networks=[torch.nn.Identity()],
                parallel=True,
                max_workers=0,
            )


if __name__ == "__main__":
    unittest.main()

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
from tests.test_utils import assert_allclose, skip_if_no_cuda

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
    """Tests for opt-in parallel ensemble inference."""

    def _get_loader(self):
        """Return a loader whose image values identify the sample index."""

        class TestDataset(torch.utils.data.Dataset):
            def __len__(self):
                """Return the number of samples."""
                return 4

            def __getitem__(self, index):
                """Return one indexed sample."""
                return {"image": torch.tensor([index]), "label": torch.zeros(1)}

        return torch.utils.data.DataLoader(TestDataset())

    def test_parallel_inference_and_output_order(self):
        """Verify concurrent execution and prediction-key ordering."""

        barrier = threading.Barrier(2, timeout=5)
        thread_ids = set()
        thread_ids_lock = threading.Lock()

        class BarrierNet(torch.nn.Module):
            def __init__(self, offset):
                """Initialize the network with an additive offset."""
                super().__init__()
                self.offset = offset

            def forward(self, x):
                """Synchronize both workers, record their threads, and add the offset."""
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
            """Verify that predictions retain network order."""
            expected_batch = torch.tensor([engine.state.iteration - 1], dtype=torch.float32)
            assert_allclose(engine.state.output[0]["pred_0"], expected_batch + 1)
            assert_allclose(engine.state.output[0]["pred_1"], expected_batch + 2)

        val_engine.run()
        self.assertEqual(len(thread_ids), 2)

    def test_parallel_run_can_be_reused(self):
        """Verify that a parallel evaluator can run more than once."""

        class AddNet(torch.nn.Module):
            def __init__(self, offset):
                """Initialize the network with an additive offset."""
                super().__init__()
                self.offset = offset

            def forward(self, x):
                """Add the configured offset to the input."""
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
            """Record each prediction from the first network."""
            iterations.append(engine.state.output[0]["pred_0"].item())

        val_engine.run()
        val_engine.run()
        self.assertEqual(iterations, [1, 2, 3, 4, 1, 2, 3, 4])

    def test_parallel_results_match_sequential_results(self):
        """Verify that parallel inference preserves sequential predictions."""

        class ScaleNet(torch.nn.Module):
            def __init__(self, scale):
                """Initialize the network with a multiplicative scale."""
                super().__init__()
                self.scale = scale

            def forward(self, x):
                """Scale the input by the configured factor."""
                return x * self.scale

        class TestDataset(torch.utils.data.Dataset):
            def __getitem__(self, index):
                """Return one indexed sample."""
                return {"image": torch.tensor([index], dtype=torch.float32), "label": torch.zeros(1)}

            def __len__(self):
                """Return the number of samples."""
                return 5

        def evaluate(parallel):
            """Evaluate the dataset with sequential or parallel network inference.

            Args:
                parallel: whether to run the networks concurrently.

            Returns:
                Prediction dictionaries recorded after each iteration.
            """
            outputs = []
            val_engine = EnsembleEvaluator(
                device=torch.device("cpu:0"),
                val_data_loader=torch.utils.data.DataLoader(TestDataset(), batch_size=1),
                networks=[ScaleNet(2), ScaleNet(3)],
                pred_keys=["double", "triple"],
                parallel=parallel,
            )

            @val_engine.on(Events.ITERATION_COMPLETED)
            def record_output(engine):
                """Record the current prediction dictionary."""
                outputs.append(engine.state.output[0])

            val_engine.run()
            return outputs

        sequential = evaluate(parallel=False)
        parallel = evaluate(parallel=True)
        self.assertEqual(len(parallel), len(sequential))
        for parallel_output, sequential_output in zip(parallel, sequential):
            self.assertEqual(parallel_output.keys(), sequential_output.keys())
            for key in sequential_output:
                assert_allclose(parallel_output[key], sequential_output[key])

    def test_parallel_inference_propagates_errors(self):
        """Verify that worker exceptions abort evaluation and clear the executor."""

        class FailingNet(torch.nn.Module):
            def forward(self, x):
                """Raise an inference error."""
                raise RuntimeError("inference failed")

        val_engine = EnsembleEvaluator(
            device=torch.device("cpu:0"),
            val_data_loader=self._get_loader(),
            networks=[torch.nn.Identity(), FailingNet()],
            parallel=True,
        )
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            val_engine.run()
        self.assertIsNone(val_engine._executor)

    def test_invalid_max_workers(self):
        """Verify that non-positive worker counts are rejected."""

        with self.assertRaisesRegex(ValueError, "max_workers"):
            EnsembleEvaluator(
                device=torch.device("cpu:0"),
                val_data_loader=self._get_loader(),
                networks=[torch.nn.Identity()],
                parallel=True,
                max_workers=0,
            )

    def test_parallel_inference_without_device(self):
        """Verify that parallel inference also works when no device is configured."""

        val_engine = EnsembleEvaluator(
            device=None,
            val_data_loader=self._get_loader(),
            networks=[torch.nn.Identity(), torch.nn.Identity()],
            parallel=True,
        )
        outputs = []

        @val_engine.on(Events.ITERATION_COMPLETED)
        def record_output(engine):
            """Record each prediction from the first network."""
            outputs.append(engine.state.output[0]["pred_0"].item())

        val_engine.run()
        self.assertEqual(outputs, [0, 1, 2, 3])

    def test_parallel_shared_network_stays_in_eval_mode(self):
        """Verify that a shared network remains in eval mode until both workers finish."""

        class SharedModeNet(torch.nn.Module):
            """Record whether a shared module is restored while another worker is running."""

            def __init__(self):
                """Initialize synchronization state."""
                super().__init__()
                self.barrier = threading.Barrier(2, timeout=5)
                self.restored = threading.Event()
                self.calls = 0
                self.observed_training = []
                self.lock = threading.Lock()

            def forward(self, x):
                """Observe training mode after giving the first worker time to restore it."""
                with self.lock:
                    self.calls += 1
                    call_number = self.calls
                self.barrier.wait()
                if call_number == 2:
                    self.restored.wait(timeout=0.2)
                with self.lock:
                    self.observed_training.append(self.training)
                return x

            def train(self, mode=True):
                """Signal when the module is restored to training mode."""
                result = super().train(mode)
                if mode:
                    self.restored.set()
                return result

        class TestDataset(torch.utils.data.Dataset):
            """One-sample dataset for the shared-network mode test."""

            def __len__(self):
                """Return the number of samples."""
                return 1

            def __getitem__(self, index):
                """Return one sample."""
                return {"image": torch.zeros(1), "label": torch.zeros(1)}

        network = SharedModeNet()
        network.train()
        val_engine = EnsembleEvaluator(
            device=torch.device("cpu:0"),
            val_data_loader=torch.utils.data.DataLoader(TestDataset()),
            networks=[network, network],
            parallel=True,
        )

        val_engine.run()
        self.assertEqual(network.observed_training, [False, False])
        self.assertTrue(network.training)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires at least two CUDA devices")
    def test_multiple_cuda_devices_are_rejected(self):
        """Verify that inputs spanning CUDA devices are rejected."""

        with self.assertRaisesRegex(ValueError, "one CUDA device"):
            EnsembleEvaluator._find_cuda_device(
                (torch.zeros(1, device="cuda:0"), {"image": torch.zeros(1, device="cuda:1")})
            )

    @skip_if_no_cuda
    def test_parallel_inference_waits_for_custom_cuda_stream(self):
        """Verify CUDA stream dependencies with an explicit or inferred device.

        ``prepare_batch`` queues a delayed fill on the caller's custom stream and returns CUDA
        tensors. Each worker must wait for that fill before running inference. The delayed
        networks then queue work on worker streams, and the caller must wait for completion
        before the iteration callback reads predictions. Observing ``(2, 3)`` verifies both
        handoffs. The subtest covers an explicit ``cuda:0`` configuration and ``device=None``,
        where the synchronization device is inferred from the prepared tensors.
        """

        cuda_device = torch.device("cuda:0")
        worker_streams = set()
        worker_devices = set()
        worker_streams_lock = threading.Lock()

        class TestDataset(torch.utils.data.Dataset):
            """One-sample dataset for the CUDA stream ordering test."""

            def __len__(self):
                """Return the number of samples."""
                return 1

            def __getitem__(self, index):
                """Return one CPU sample."""
                return {"image": torch.zeros(1), "label": torch.zeros(1)}

        def prepare_batch(batchdata, device, non_blocking, **kwargs):
            """Produce input after delayed caller-stream work.

            Args:
                batchdata: CPU batch returned by the data loader.
                device: configured evaluator device, or ``None`` when inference is inferred.
                non_blocking: whether device transfers should be non-blocking.
                kwargs: additional transfer options.

            Returns:
                CUDA image and label tensors for network inference.
            """

            target_device = device if device is not None else torch.device("cuda:0")
            image = torch.zeros_like(batchdata["image"], device=target_device)
            torch.cuda._sleep(100_000_000)
            image.fill_(1)
            return image, batchdata["label"].to(device=target_device)

        class DelayedAddNet(torch.nn.Module):
            """Add an offset after delayed work on the worker CUDA stream."""

            def __init__(self, offset):
                """Initialize the network with an additive offset."""
                super().__init__()
                self.offset = offset

            def forward(self, x):
                """Add the offset after delaying worker-stream work."""
                with worker_streams_lock:
                    worker_streams.add(torch.cuda.current_stream(cuda_device).cuda_stream)
                    worker_devices.add(torch.cuda.current_device())
                torch.cuda._sleep(50_000_000)
                return x + self.offset

        for device in (cuda_device, None):
            with self.subTest(device=device):
                with worker_streams_lock:
                    worker_streams.clear()
                    worker_devices.clear()
                val_engine = EnsembleEvaluator(
                    device=device,
                    val_data_loader=torch.utils.data.DataLoader(TestDataset()),
                    networks=[DelayedAddNet(1), DelayedAddNet(2)],
                    parallel=True,
                    prepare_batch=prepare_batch,
                )
                observed = []

                @val_engine.on(Events.ITERATION_COMPLETED)
                def record_output(engine):
                    """Force the caller to consume results before the iteration completes."""

                    observed.append((engine.state.output[0]["pred_0"].item(), engine.state.output[0]["pred_1"].item()))

                with torch.cuda.stream(torch.cuda.Stream(device=cuda_device)):
                    val_engine.run()
                self.assertEqual(observed, [(2, 3)])
                self.assertEqual(worker_devices, {cuda_device.index})
                self.assertEqual(len(worker_streams), 2)
                self.assertNotIn(torch.cuda.default_stream(cuda_device).cuda_stream, worker_streams)


if __name__ == "__main__":
    unittest.main()

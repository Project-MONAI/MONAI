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

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from torch.utils.data import IterableDataset as TorchIterableDataset

from monai.data import DataLoader, IterableDataset, ShuffleBuffer
from monai.data import iterable_dataset as iterable_dataset_module
from monai.utils import convert_data_type


class _UnshardedMonaiIterable(IterableDataset):
    """MONAI iterable subclass that intentionally does not partition itself."""

    def __iter__(self):
        yield from self.data


class _WorkerShardedTorchIterable(TorchIterableDataset):
    """PyTorch iterable source that partitions itself across workers."""

    def __init__(self, size):
        self.size = size

    def __iter__(self):
        worker_info = iterable_dataset_module.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        yield from range(worker_id, self.size, num_workers)


class TestShuffleBuffer(unittest.TestCase):
    def test_shape(self):
        buffer = ShuffleBuffer([1, 2, 3, 4], seed=0)
        num_workers = 2 if sys.platform == "linux" else 0
        dataloader = DataLoader(
            dataset=buffer, batch_size=2, num_workers=num_workers, persistent_workers=num_workers > 0
        )
        output = [convert_data_type(x, np.ndarray)[0] for x in dataloader]
        buffer.seed += 1
        output2 = [convert_data_type(x, np.ndarray)[0] for x in dataloader]  # test repeating
        if num_workers == 0:
            np.testing.assert_allclose(output, [[2, 1], [3, 4]])
            np.testing.assert_allclose(output2, [[3, 1], [2, 4]])
        else:  # multiprocess shuffle
            np.testing.assert_allclose(output, [[2, 3], [1, 4]], err_msg=f"seed {buffer.seed}")
            np.testing.assert_allclose(output2, [[1, 4], [2, 3]], err_msg=f"seed {buffer.seed}")

    def test_monai_iterable_source_is_detected_as_worker_sharded(self):
        """Verify MONAI iterable sources avoid a second worker partition by default."""
        outputs = []
        for worker_id in range(2):
            source = IterableDataset(range(40))
            buffer = ShuffleBuffer(source, buffer_size=8, seed=7)
            worker_info = SimpleNamespace(num_workers=2, id=worker_id)
            with patch("monai.data.iterable_dataset.get_worker_info", return_value=worker_info):
                outputs.extend(buffer)

        self.assertEqual(len(outputs), 40)
        self.assertEqual(set(outputs), set(range(40)))

    def test_worker_sharded_source_is_not_sharded_twice(self):
        """Verify an explicitly worker-sharded source is not repartitioned."""
        sources = [IterableDataset(range(40)), _WorkerShardedTorchIterable(40)]
        for source in sources:
            outputs = []
            for worker_id in range(2):
                buffer = ShuffleBuffer(
                    source, transform=lambda item: item + 40, buffer_size=8, seed=7, source_shards_by_worker=True
                )
                worker_info = SimpleNamespace(num_workers=2, id=worker_id)
                with patch("monai.data.iterable_dataset.get_worker_info", return_value=worker_info):
                    outputs.extend(buffer)

            self.assertEqual(len(outputs), 40)
            self.assertEqual(set(outputs), set(range(40, 80)))

    def test_explicit_unsharded_source_keeps_outer_worker_partition(self):
        """Verify explicit unsharded mode preserves outer worker partitioning."""
        outputs = []
        for worker_id in range(2):
            source = _UnshardedMonaiIterable(range(40))
            buffer = ShuffleBuffer(source, buffer_size=8, seed=7, source_shards_by_worker=False)
            worker_info = SimpleNamespace(num_workers=2, id=worker_id)
            with patch("monai.data.iterable_dataset.get_worker_info", return_value=worker_info):
                outputs.extend(buffer)

        self.assertEqual(len(outputs), 40)
        self.assertEqual(set(outputs), set(range(40)))

    def test_epochs(self):
        buffer = ShuffleBuffer([1, 2, 3, 4], seed=0, epochs=2)
        output = [convert_data_type(x, np.ndarray)[0] for x in DataLoader(dataset=buffer, batch_size=2)]
        np.testing.assert_allclose(output, [[2, 1], [3, 4], [4, 2], [3, 1]])


if __name__ == "__main__":
    unittest.main()

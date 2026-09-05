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
import types
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data import CacheDataset, DataLoader, Dataset, ZipDataset
from monai.data.utils import set_rnd
from monai.transforms import Compose, DataStatsd, Randomizable, SimulateDelayd
from monai.utils import convert_to_numpy, set_determinism
from tests.test_utils import assert_allclose

TEST_CASE_1 = [[{"image": np.asarray([1, 2, 3])}, {"image": np.asarray([4, 5])}]]

TEST_CASE_2 = [[{"label": torch.as_tensor([[3], [2]])}, {"label": np.asarray([[1], [2]])}]]

_CYCLIC_DATASET_SIZE = 4
_CYCLIC_BATCH_SIZE = 1
_CYCLIC_NUM_WORKERS = 0
_CYCLIC_TEST_SEED = 42


class TestDataLoader(unittest.TestCase):
    def test_values(self):
        datalist = [
            {"image": "spleen_19.nii.gz", "label": "spleen_label_19.nii.gz"},
            {"image": "spleen_31.nii.gz", "label": "spleen_label_31.nii.gz"},
        ]
        transform = Compose(
            [
                DataStatsd(keys=["image", "label"], data_shape=False, value_range=False, data_value=True),
                SimulateDelayd(keys=["image", "label"], delay_time=0.1),
            ]
        )
        dataset = CacheDataset(data=datalist, transform=transform, cache_rate=0.5, cache_num=1)
        n_workers = 0 if sys.platform == "win32" else 2
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=n_workers)
        for d in dataloader:
            self.assertEqual(d["image"][0], "spleen_19.nii.gz")
            self.assertEqual(d["image"][1], "spleen_31.nii.gz")
            self.assertEqual(d["label"][0], "spleen_label_19.nii.gz")
            self.assertEqual(d["label"][1], "spleen_label_31.nii.gz")

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_exception(self, datalist):
        dataset = Dataset(data=datalist, transform=None)
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=0)
        with self.assertRaisesRegex((TypeError, RuntimeError), "Collate error on the key"):
            for _ in dataloader:
                pass


class _RandomDataset(torch.utils.data.Dataset, Randomizable):
    def __getitem__(self, index):
        return self.R.randint(0, 1000, (1,))

    def __len__(self):
        return 8


class TestLoaderRandom(unittest.TestCase):
    """
    Testing data loader working with the randomizable interface
    """

    def setUp(self):
        set_determinism(0)

    def tearDown(self):
        set_determinism(None)

    @parameterized.expand([[1], [0]])
    def test_randomize(self, workers):
        set_determinism(0)
        dataset = _RandomDataset()
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=workers)
        output = []
        for _ in range(1):  # need persistent workers for reproducibility of num_workers 0, 1
            for batch in dataloader:
                output.extend(batch.data.numpy().flatten().tolist())
        set_determinism(None)
        self.assertListEqual(output, [594, 170, 292, 589, 153, 811, 21, 550])

    def test_zipdataset(self):
        dataset = ZipDataset([_RandomDataset(), ZipDataset([_RandomDataset(), _RandomDataset()])])
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
        output = []
        for _ in range(2):
            for batch in dataloader:
                output.extend([convert_to_numpy(batch, wrap_sequence=False)])
        assert_allclose(np.stack(output).flatten()[:7], np.array([594, 170, 594, 170, 594, 170, 524]))


class _CyclicConfigDataset(torch.utils.data.Dataset):
    """
    Dataset holding an attribute whose object graph contains a reference cycle.

    This mirrors OmegaConf/Hydra configs, whose child nodes hold a back-reference
    to their parent node. Seeding such a dataset used to recurse forever in
    ``monai.data.utils.set_rnd`` (see issue #8087).
    """

    def __init__(self):
        parent = types.SimpleNamespace()
        child = types.SimpleNamespace()
        parent.child = child
        child.parent = parent  # reference cycle, as in an OmegaConf parent/child graph
        self.cfg = parent

    def __len__(self):
        return _CYCLIC_DATASET_SIZE

    def __getitem__(self, index):
        return torch.tensor([index])


class _SeedRecorder:
    def __init__(self):
        self.seed = None

    def set_random_state(self, seed):
        self.seed = seed


class TestLoaderRecursion(unittest.TestCase):
    def test_cyclic_reference_no_recursion(self):
        # Constructing the loader seeds the dataset (num_workers=0). A reference cycle in the
        # dataset's attributes must not raise RecursionError while walking the object graph.
        dataloader = DataLoader(
            _CyclicConfigDataset(), batch_size=_CYCLIC_BATCH_SIZE, num_workers=_CYCLIC_NUM_WORKERS, shuffle=False
        )
        self.assertEqual(len(list(dataloader)), _CYCLIC_DATASET_SIZE)

    def test_cyclic_list_reference_no_recursion(self):
        """Test seeding a dataset whose configuration list contains itself."""
        dataset = _CyclicConfigDataset()
        dataset.cfg = []
        dataset.cfg.append(dataset.cfg)
        dataloader = DataLoader(dataset, batch_size=_CYCLIC_BATCH_SIZE, num_workers=_CYCLIC_NUM_WORKERS, shuffle=False)
        self.assertEqual(len(list(dataloader)), _CYCLIC_DATASET_SIZE)

    def test_cyclic_list_preserves_seed_advancement(self):
        """Test a cyclic list does not erase seed advancement from an earlier item."""
        dataset = _CyclicConfigDataset()
        nested_randomizable = _SeedRecorder()
        following_randomizable = _SeedRecorder()
        dataset.cfg = [nested_randomizable]
        dataset.cfg.append(dataset.cfg)
        dataset.following_randomizable = following_randomizable

        set_rnd(dataset, seed=_CYCLIC_TEST_SEED)

        self.assertEqual(nested_randomizable.seed, _CYCLIC_TEST_SEED)
        self.assertEqual(following_randomizable.seed, _CYCLIC_TEST_SEED + 1)


if __name__ == "__main__":
    unittest.main()

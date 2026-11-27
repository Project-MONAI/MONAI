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
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist

from monai.data import DistributedWeightedRandomSampler
from tests.test_utils import DistCall, DistTestCase, skip_if_darwin, skip_if_windows

NUM_RANKS = 2


@DistCall(nnodes=1, nproc_per_node=NUM_RANKS)
def run_weighted_samplers(output_dir):
    data = [1, 2, 3, 4, 5]
    weights = [1, 2, 3, 4, 5]
    samplers = {
        "sampling": DistributedWeightedRandomSampler(
            weights=weights, dataset=data, shuffle=False, generator=torch.Generator().manual_seed(0)
        ),
        "num_samples": DistributedWeightedRandomSampler(
            weights=weights,
            num_samples_per_rank=5,
            dataset=data,
            shuffle=False,
            generator=torch.Generator().manual_seed(123),
        ),
    }
    samples = {name: [data[i] for i in sampler] for name, sampler in samplers.items()}
    Path(output_dir, f"{dist.get_rank()}.json").write_text(json.dumps(samples))


@skip_if_windows
@skip_if_darwin
class DistributedWeightedRandomSamplerTest(DistTestCase):
    @classmethod
    def setUpClass(cls):
        # one process group computes every scenario; each test checks its own result in the parent
        with tempfile.TemporaryDirectory() as tempdir:
            run_weighted_samplers(tempdir)
            cls.samples = [json.loads(Path(tempdir, f"{rank}.json").read_text()) for rank in range(NUM_RANKS)]

    def test_sampling(self):
        self.assertEqual([rank["sampling"] for rank in self.samples], [[5, 5, 5], [1, 4, 4]])

    def test_num_samples(self):
        self.assertEqual([rank["num_samples"] for rank in self.samples], [[3, 1, 5, 1, 5], [4, 2, 4, 2, 4]])


if __name__ == "__main__":
    unittest.main()

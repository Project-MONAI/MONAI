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

import unittest

import torch
import numpy as np
import torch.distributed as dist

from monai.data import CacheDataset, DataLoader, DistributedSampler
from monai.transforms import ToTensor
from tests.utils import DistCall, DistTestCase


class DistributedSamplerTest(DistTestCase):
    @DistCall(nnodes=1, nproc_per_node=2)
    def test_even(self):
        data = [1, 2, 3, 4, 5]
        sampler = DistributedSampler(dataset=data, shuffle=False)
        samples = np.array([data[i] for i in list(sampler)])
        self.assertEqual(dist.get_rank(), sampler.rank)
        if dist.get_rank() == 0:
            np.testing.assert_allclose(samples, np.array([1, 3, 5]))

        if dist.get_rank() == 1:
            np.testing.assert_allclose(samples, np.array([2, 4, 1]))

    @DistCall(nnodes=1, nproc_per_node=2)
    def test_uneven(self):
        data = [1, 2, 3, 4, 5]
        sampler = DistributedSampler(dataset=data, shuffle=False, even_divisible=False)
        samples = np.array([data[i] for i in list(sampler)])
        self.assertEqual(dist.get_rank(), sampler.rank)
        if dist.get_rank() == 0:
            np.testing.assert_allclose(samples, np.array([1, 3, 5]))

        if dist.get_rank() == 1:
            np.testing.assert_allclose(samples, np.array([2, 4]))

    @DistCall(nnodes=1, nproc_per_node=2)
    def test_cachedataset(self):
        data = [1, 2, 3, 4, 5]
        dataset = CacheDataset(data=data, transform=ToTensor(track_meta=False), cache_rate=1.0, runtime_cache=True)
        sampler = DistributedSampler(dataset=dataset, shuffle=False)
        dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=1, num_workers=2)
        for i in range(3):
            if i > 0:
                # verify the runtime cache content
                for j, c in enumerate(dataset._cache):
                    self.assertTrue(isinstance(c, torch.Tensor))
                    torch.testing.assert_allclose(c, j + 1)
            for d in dataloader:
                if dist.get_rank() == 0:
                    print("rank 0: ", d[0])

                if dist.get_rank() == 1:
                    print("rank 1: ", d[0])


if __name__ == "__main__":
    unittest.main()

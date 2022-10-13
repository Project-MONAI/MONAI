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

import os
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from parameterized import parameterized
from torch.utils.data.distributed import DistributedSampler

from monai.data import DataLoader, SharedCacheDataset
from monai.transforms import Compose, LoadImaged, RandomizableTransform, Transform

TEST_CASE_1 = [Compose([LoadImaged(keys=["image", "label", "extra"])]), (128, 128, 128)]
TEST_CASE_2 = [None, (128, 128, 128)]


class TestCacheDataset(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, transform, expected_shape):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=[128, 128, 128]), np.eye(4), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tempdir:
            test_data = []
            for i in ["1", "2"]:
                for k in ["image", "label", "extra"]:
                    nib.save(test_image, os.path.join(tempdir, f"{k}{i}.nii.gz"))
                test_data.append({k: os.path.join(tempdir, f"{k}{i}.nii.gz") for k in ["image", "label", "extra"]})

            dataset = SharedCacheDataset(data=test_data, transform=transform)
            data1 = dataset[0]
            data2 = dataset[1]
            data3 = dataset[0:-1]
            data4 = dataset[-1]
            self.assertEqual(len(data3), 1)

            if transform is None:
                # Check without providing transfrom
                dataset2 = SharedCacheDataset(data=test_data)
                for k in ["image", "label", "extra"]:
                    self.assertEqual(dataset[0][k], dataset2[0][k])

        if transform is None:
            self.assertEqual(data1["image"], os.path.join(tempdir, "image1.nii.gz"))
            self.assertEqual(data2["label"], os.path.join(tempdir, "label2.nii.gz"))
            self.assertEqual(data4["image"], os.path.join(tempdir, "image2.nii.gz"))
        else:
            self.assertTupleEqual(data1["image"].shape, expected_shape)
            self.assertTupleEqual(data1["label"].shape, expected_shape)
            self.assertTupleEqual(data1["extra"].shape, expected_shape)
            self.assertTupleEqual(data2["image"].shape, expected_shape)
            self.assertTupleEqual(data2["label"].shape, expected_shape)
            self.assertTupleEqual(data2["extra"].shape, expected_shape)
            for d in data3:
                self.assertTupleEqual(d["image"].shape, expected_shape)


class TransformNonrandom(Transform):
    def __call__(self, x):
        return np.array([x * 10])


class TransformRandom(RandomizableTransform):
    def __call__(self, x):
        return x + 1


def main_worker(rank, nprocs, cache_list):

    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 1

    device = torch.device(rank) if has_cuda else torch.device("cpu")
    device_ids = [rank] if has_cuda else None
    output_device = device if has_cuda else None
    backend = "nccl" if has_cuda else "gloo"

    dist.init_process_group(backend=backend, init_method="tcp://127.0.0.1:12345", world_size=nprocs, rank=rank)
    model = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, bias=True).to(device=device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids, output_device=output_device, find_unused_parameters=False
    )

    data_list1 = list(range(4 * nprocs))
    transform = Compose([TransformNonrandom(), TransformRandom()])

    dataset = SharedCacheDataset(data=data_list1, transform=transform, copy_cache=False, cache_list=cache_list)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, num_workers=2, sampler=sampler)
    ids = list(range(rank, len(data_list1), nprocs))  # ids for given process that DistributedSampler will use

    # each process goes only over a small subset of the data (and caches in shared cache)
    p = 0
    for i, d in enumerate(dataloader):
        # print(rank, i, d)
        expected_data = data_list1[ids[i]] * 10 + 1
        np.testing.assert_allclose([[expected_data]], d)
        p = p + 1
    assert p == len(dataset) // nprocs, f"each process processed {p} out of {len(dataset)}"
    torch.distributed.barrier()

    # at this point the full dataset is cached, and every process has access to it
    # lets inspect cache
    for i in range(len(dataset)):
        expected_data = data_list1[i] * 10  # cached part was only the first transform
        cache = dataset._cache[i]
        np.testing.assert_allclose(expected_data, cache)
    torch.distributed.barrier()

    # lets update cache directly by +1
    for i in ids:
        dataset._cache[i] += 1
    torch.distributed.barrier()

    # inspect results, must have output +1 (since cache will be used instead of the first transform)
    for i, d in enumerate(dataloader):
        expected_data = data_list1[rank : len(data_list1) : nprocs][i] * 10 + 1 + 1  # expecting +1 in output
        # print(rank, i, d, expected_data)
        np.testing.assert_allclose([[expected_data]], d)
    torch.distributed.barrier()

    # print('processed rank', rank)
    cache_list[:] = []
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class TestDDP(unittest.TestCase):
    def test_ddp_ops(self):
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            nprocs = torch.cuda.device_count()
        else:
            nprocs = 2

        manager = torch.multiprocessing.Manager()
        cache_list = manager.list()
        torch.multiprocessing.spawn(main_worker, nprocs=nprocs, args=(nprocs, cache_list))


if __name__ == "__main__":
    unittest.main()

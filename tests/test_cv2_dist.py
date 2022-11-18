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
import torch.distributed as dist
from torch.cuda.amp import autocast

# FIXME: test for the workaround of https://github.com/Project-MONAI/MONAI/issues/5291
from monai.config.deviceconfig import print_config
from tests.utils import skip_if_no_cuda


def main_worker(rank, ngpus_per_node):
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:12345", world_size=ngpus_per_node, rank=rank)
    # `benchmark = True` is not compatible with openCV in PyTorch 22.09 docker for multi-gpu training
    torch.backends.cudnn.benchmark = True

    model = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, bias=True).to(rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=False
    )
    x = torch.ones(1, 1, 12, 12, 12).to(rank)
    with autocast(enabled=True):
        model(x)


@skip_if_no_cuda
class TestCV2Dist(unittest.TestCase):
    def test_cv2_cuda_ops(self):
        print_config()
        ngpus_per_node = torch.cuda.device_count()
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))


if __name__ == "__main__":
    unittest.main()

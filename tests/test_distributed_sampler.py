# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.distributed as dist

from monai.data import DistributedSampler


def test(expected, **kwargs):
    dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(dist.get_rank())
    data = [1, 2, 3, 4, 5]
    sampler = DistributedSampler(dataset=data, **kwargs)
    samples = np.array([data[i] for i in list(sampler)])
    if dist.get_rank() == 0:
        np.testing.assert_allclose(samples, np.array(expected[0]))

    if dist.get_rank() == 1:
        np.testing.assert_allclose(samples, np.array(expected[1]))

    dist.destroy_process_group()


def main():
    test(shuffle=False, expected=[[1, 3, 5], [2, 4, 1]])
    test(shuffle=False, even_divisible=False, expected=[[1, 3, 5], [2, 4]])


# suppose to execute on 2 rank processes
# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
#        --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
#        --master_addr="localhost" --master_port=1234
#        test_distributed_sampler.py

if __name__ == "__main__":
    main()

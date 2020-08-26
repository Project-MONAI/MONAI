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
import torch.distributed as dist

from monai.data import partition_dataset

TEST_CASES = [
    [False, 0, np.array([0, 2, 4, 6, 8]), np.array([1, 3, 5, 7, 9])],
    [True, 0, np.array([4, 7, 3, 0, 6]), np.array([1, 5, 9, 8, 2])],
    [True, 100, np.array([0, 5, 3, 4, 7]), np.array([8, 2, 6, 1, 9])],
]


def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    data = list(range(10))

    for case in TEST_CASES:
        data_part = partition_dataset(data, shuffle=case[0], seed=case[1])

        if dist.get_rank() == 0:
            np.testing.assert_allclose(np.array(data_part), case[2])

        if dist.get_rank() == 1:
            np.testing.assert_allclose(np.array(data_part), case[3])

    dist.destroy_process_group()


# suppose to execute on 2 rank processes
# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
#        --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
#        --master_addr="192.168.1.1" --master_port=1234
#        test_partition_dataset.py

if __name__ == "__main__":
    main()

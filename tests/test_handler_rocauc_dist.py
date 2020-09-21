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

from monai.handlers import ROCAUC


def main():
    dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(dist.get_rank())
    auc_metric = ROCAUC(to_onehot_y=True, softmax=True)

    if dist.get_rank() == 0:
        y_pred = torch.tensor([[0.1, 0.9], [0.3, 1.4]], device=torch.device("cuda:0"))
        y = torch.tensor([[0], [1]], device=torch.device("cuda:0"))
        auc_metric.update([y_pred, y])

    if dist.get_rank() == 1:
        y_pred = torch.tensor([[0.2, 0.1], [0.1, 0.5]], device=torch.device("cuda:1"))
        y = torch.tensor([[0], [1]], device=torch.device("cuda:1"))
        auc_metric.update([y_pred, y])

    result = auc_metric.compute()
    np.testing.assert_allclose(0.75, result)

    dist.destroy_process_group()


# suppose to execute on 2 rank processes
# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_PER_NODE
#        --nnodes=NUM_NODES --node_rank=INDEX_CURRENT_NODE
#        --master_addr="192.168.1.1" --master_port=1234
#        test_handler_rocauc_dist.py

if __name__ == "__main__":
    main()

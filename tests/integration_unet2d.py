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

import sys
from functools import partial

import torch
import torch.nn as nn
import numpy as np

from ignite.engine import Events, create_supervised_trainer

from monai import application, data, networks, utils
import monai.data.augments.augments as augments


def run_test(batch_size = 64, train_steps = 100, device = torch.device("cuda:0")):
    def generate_test_batch():
        for _ in range(train_steps):
            im, seg = utils.generateddata.create_test_image_2d(128, 128, noise_max=1, num_objs=4, num_seg_classes=1)
            yield im[None], seg[None].astype(np.float32)
            

    def _prepare_batch(batch, device=None, non_blocking=False):
        x, y = batch
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


    net = networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        num_classes=1,
        channels=(4, 8, 16, 32),
        strides=(2, 2, 2),
        num_res_units=2,
    )

    loss = networks.losses.DiceLoss()
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    src = data.streams.BatchStream(generate_test_batch(), batch_size)

    loss_fn = lambda i, j: loss(i[0], j)

    trainer = create_supervised_trainer(net, opt, loss_fn, device, False, _prepare_batch)

    trainer.run(src,1)

    return trainer.state.output


if __name__ == '__main__':
    result = run_test()
    
    sys.exit(0 if result < 1 else 1)
    
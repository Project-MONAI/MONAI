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

from ignite.engine import Events


class LrScheduleHander:
    """
    ignite handler to update the Learning Rate based on PyTorch LR scheduler.
    """

    def __init__(self, lr_scheduler):
        """
        Args:
            lr_scheduler (torch.optim.lr_scheduler): typically, lr_scheduler should be PyTorch
                lr_scheduler object. if customized version, must have `step` and `get_last_lr` methods.

        """
        self.lr_scheduler = lr_scheduler

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self)

    def __call__(self, engine):
        self.lr_scheduler.step()
        print(f'Update learning rate to: {self.lr_scheduler.get_last_lr()[0]}')

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

from collections import defaultdict

import monai
from ignite.engine import Events


@monai.utils.export("monai.handlers")
@monai.utils.alias("metriclogger")
class MetricLogger:

    def __init__(self, loss_transform=lambda x: x, metric_transform=lambda x: x):
        self.loss_transform = loss_transform
        self.metric_transform = metric_transform
        self.loss = []
        self.metrics = defaultdict(list)

    def attach(self, engine):
        return engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine):
        self.loss.append(self.loss_transform(engine.state.output))

        for m, v in engine.state.metrics.items():
            v = self.metric_transform(v)
            #             # metrics may not be added on the first timestep, pad the list if this is the case
            #             # so that each metric list is the same length as self.loss
            #             if len(self.metrics[m])==0:
            #                 self.metrics[m].append([v[0]]*len(self.loss))

            self.metrics[m].append(v)

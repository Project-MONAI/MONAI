from collections import defaultdict

import monai


@monai.utils.export("monai.application.handlers")
@monai.utils.alias("metriclogger")
class MetricLogger:

    def __init__(self, loss_transform=lambda x: x, metric_transform=lambda x: x):
        self.loss_transform = loss_transform
        self.metric_transform = metric_transform
        self.loss = []
        self.metrics = defaultdict(list)

    def attach(self, engine):
        return engine.add_event_handler(monai.application.engine.Events.ITERATION_COMPLETED, self)

    def __call__(self, engine):
        self.loss.append(self.loss_transform(engine.state.output))

        for m, v in engine.state.metrics.items():
            v = self.metric_transform(v)
            #             # metrics may not be added on the first timestep, pad the list if this is the case
            #             # so that each metric list is the same length as self.loss
            #             if len(self.metrics[m])==0:
            #                 self.metrics[m].append([v[0]]*len(self.loss))

            self.metrics[m].append(v)

import unittest

from torch.nn import Module
from ignite.engine import Engine, Events

from monai.handlers.parameter_scheduler import ParamSchedulerHandler
from ignite.engine import Events


class ToyNet(Module):
    def __init__(self, value):
        super(ToyNet, self).__init__()
        self.value = value

    def forward(self, input):
        return input

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value


net = ToyNet(value=1)
engine = Engine(lambda e, b: None)
ParamSchedulerHandler(
    parameter_setter=net.get_value,
    value_calculator="linear",
    vc_kwargs={"initial_value": 1, "step_constant": 2, "step_max_value": 4, "max_value": 5},
    epoch_level=True,
    event=Events.EPOCH_COMPLETED,
).attach(engine)
engine.run([0] * 8, max_epochs=5)

print(net.get_value())

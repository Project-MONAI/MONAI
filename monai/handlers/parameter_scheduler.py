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

import logging
from bisect import bisect_right
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

from monai.config import IgniteInfo
from monai.utils import min_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine, Events
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")


class ParamSchedulerHandler:
    """
    General purpose scheduler for parameters values. By default it can schedule in a linear, exponential, step or
    multistep function. One can also pass Callables to have customized scheduling logic.

    Args:
        parameter_setter (Callable): Function that sets the required parameter
        value_calculator (Union[str,Callable]): Either a string ('linear', 'exponential', 'step' or 'multistep')
         or Callable for custom logic.
        vc_kwargs (Dict): Dictionary that stores the required parameters for the value_calculator.
        epoch_level (bool): Whether the step is based on epoch or iteration. Defaults to False.
        name (Optional[str]): Identifier of logging.logger to use, if None, defaulting to ``engine.logger``.
        event (Optional[str]): Event to which the handler attaches. Defaults to Events.ITERATION_COMPLETED.
    """

    def __init__(
        self,
        parameter_setter: Callable,
        value_calculator: Union[str, Callable],
        vc_kwargs: Dict,
        epoch_level: bool = False,
        name: Optional[str] = None,
        event=None,
    ):
        self.epoch_level = epoch_level
        self.event = event if event is not None else Events.ITERATION_COMPLETED

        self._calculators = {
            "linear": self._linear,
            "exponential": self._exponential,
            "step": self._step,
            "multistep": self._multistep,
        }

        self._parameter_setter = parameter_setter
        self._vc_kwargs = vc_kwargs
        self._value_calculator = self._get_value_calculator(value_calculator=value_calculator)

        self.logger = logging.getLogger(name)
        self._name = name

    def _get_value_calculator(self, value_calculator):
        if isinstance(value_calculator, str):
            return self._calculators[value_calculator]
        if callable(value_calculator):
            return value_calculator
        raise ValueError(
            f"value_calculator must be either a string from {list(self._calculators.keys())} or a Callable."
        )

    def __call__(self, engine: Engine):
        if self.epoch_level:
            self._vc_kwargs["current_step"] = engine.state.epoch
        else:
            self._vc_kwargs["current_step"] = engine.state.iteration

        new_value = self._value_calculator(**self._vc_kwargs)
        self._parameter_setter(new_value)

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine that is used for training.
        """
        if self._name is None:
            self.logger = engine.logger
        engine.add_event_handler(self.event, self)

    @staticmethod
    def _linear(
        initial_value: float, step_constant: int, step_max_value: int, max_value: float, current_step: int
    ) -> float:
        """
        Keeps the parameter value to zero until step_zero steps passed and then linearly increases it to 1 until an
        additional step_one steps passed. Continues the trend until it reaches max_value.

        Args:
            initial_value (float): Starting value of the parameter.
            step_constant (int): Step index until parameter's value is kept constant.
            step_max_value (int): Step index at which parameter's value becomes max_value.
            max_value (float): Max parameter value.
            current_step (int): Current step index.

        Returns:
            float: new parameter value
        """
        if current_step <= step_constant:
            delta = 0.0
        elif current_step > step_max_value:
            delta = max_value - initial_value
        else:
            delta = (max_value - initial_value) / (step_max_value - step_constant) * (current_step - step_constant)

        return initial_value + delta

    @staticmethod
    def _exponential(initial_value: float, gamma: float, current_step: int) -> float:
        """
        Decays the parameter value by gamma every step.

        Based on the closed form of ExponentialLR from Pytorch:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html.

        Args:
            initial_value (float): Starting value of the parameter.
            gamma (float): Multiplicative factor of parameter value decay.
            current_step (int): Current step index.

        Returns:
            float: new parameter value
        """
        return initial_value * gamma**current_step

    @staticmethod
    def _step(initial_value: float, gamma: float, step_size: int, current_step: int) -> float:
        """
        Decays the parameter value by gamma every step_size.

        Based on StepLR from Pytorch:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html.

        Args:
            initial_value (float): Starting value of the parameter.
            gamma (float): Multiplicative factor of parameter value decay.
            step_size (int): Period of parameter value decay.
            current_step (int): Current step index.

        Returns
            float: new parameter value
        """
        return initial_value * gamma ** (current_step // step_size)

    @staticmethod
    def _multistep(initial_value: float, gamma: float, milestones: List[int], current_step: int) -> float:
        """
        Decays the parameter value by gamma once the number of steps reaches one of the milestones.

        Based on MultiStepLR from Pytorch.
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html.

        Args:
            initial_value (float): Starting value of the parameter.
            gamma (float): Multiplicative factor of parameter value decay.
            milestones (List[int]): List of step indices. Must be increasing.
            current_step (int): Current step index.

        Returns:
            float: new parameter value
        """
        return initial_value * gamma ** bisect_right(milestones, current_step)

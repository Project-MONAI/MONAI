# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Wrapper around NVIDIA Tools Extension for profiling MONAI ignite workflow
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union

from monai.config import IgniteInfo
from monai.utils import ensure_tuple, min_version, optional_import

_nvtx, _ = optional_import("torch._C._nvtx", descriptor="NVTX is not installed. Are you sure you have a CUDA build?")
if TYPE_CHECKING:
    from ignite.engine import Engine, Events
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")


__all__ = ["RangeHandler", "RangePushHandler", "RangePopHandler", "MarkHandler"]


class RangeHandler:
    """
    Attach a NVTX range to a pair of Ignite events.
    It pushes an NVTX range at the first event and pops it at the second event.
    Stores zero-based depth of the range that is started.

    Args:
        events: a string, pair of Ignite events, pair of Ignite event literals, or pair of Ignite events and literals.
            If a single string is provided, it should  describe the base name of a pair of default Ignite events
            with _STARTED and _COMPLETED postfix (like "EPOCH" for Events.EPOCH_STARTED and Events.EPOCH_COMPLETED).
            The accepted events are: BATCH, ITERATION, EPOCH, and ENGINE.
            If pair of literals, each should be the literal equivalent of an Ignite event, fo instance:
            ("EPOCH_STARTED" and "EPOCH_COMPLETED").
            One can combine events and literals, like (Events.EPOCH_STARTED and "EPOCH_COMPLETED").
            For the complete list of Events,
            check https://pytorch.org/ignite/generated/ignite.engine.events.Events.html.

        msg: ASCII message to associate with range.
            If not provided, the name of first event will be assigned to the NVTX range.
    """

    def __init__(
        self,
        events: Union[str, Tuple[Union[str, Events], Union[str, Events]]],
        msg: Optional[str] = None,
    ) -> None:
        self.events = self.resolve_events(events)
        if msg is None:
            if isinstance(events, str):
                # assign the prefix of the events
                msg = events
            else:
                # combine events' names
                msg = "/".join([e.name for e in self.events])
        self.msg = msg
        self.depth = None

    def resolve_events(self, events: Union[str, Tuple]) -> Tuple[Events, Events]:
        """
        Resolve the input events to create a pair of Ignite events
        """
        events = ensure_tuple(events)
        if len(events) == 1:
            return self.create_paired_events(events[0])
        if len(events) == 2:
            return (
                self.get_event(events[0]),
                self.get_event(events[1]),
            )
        raise ValueError(f"Exactly two Ignite events should be provided [received {len(events)}].")

    def create_paired_events(self, event: str) -> Tuple[Events, Events]:
        """
        Create pair of Ignite events from a event prefix name
        """
        event = event.upper()
        event_prefix = {
            "": "",
            "ENGINE": "",
            "EPOCH": "EPOCH_",
            "ITERATION": "ITERATION_",
            "BATCH": "GET_BATCH_",
        }
        return (
            self.get_event(event_prefix[event] + "STARTED"),
            self.get_event(event_prefix[event] + "COMPLETED"),
        )

    def get_event(self, event: Union[str, Events]) -> Events:
        if isinstance(event, str):
            event = event.upper()
        return Events[event]

    def attach(self, engine: Engine) -> None:
        """
        Attach an NVTX Range to specific Ignite events
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(self.events[0], self.range_push)
        engine.add_event_handler(self.events[1], self.range_pop)

    def range_push(self):
        self.depth = _nvtx.rangePushA(self.msg)

    def range_pop(self):
        _nvtx.rangePop()


class RangePushHandler:
    """
    At a specific event, pushes a range onto a stack of nested range span.
    Stores zero-based depth of the range that is started.

    Args:
        msg: ASCII message to associate with range
    """

    def __init__(self, event: Events, msg: Optional[str] = None) -> None:
        if isinstance(event, str):
            event = event.upper()
        self.event = Events[event]
        if msg is None:
            msg = self.event.name
        self.msg = msg
        self.depth = None

    def attach(self, engine: Engine) -> None:
        """
        Push an NVTX range at a specific Ignite event
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(self.event, self.range_push)

    def range_push(self):
        self.depth = _nvtx.rangePushA(self.msg)


class RangePopHandler:
    """
    At a specific event, pop a previously pushed range.
    Stores zero-based depth of the range that is started.

    Args:
        msg: ASCII message to associate with range
    """

    def __init__(self, event: Events) -> None:
        if isinstance(event, str):
            event = event.upper()
        self.event = Events[event]

    def attach(self, engine: Engine) -> None:
        """
        Pop an NVTX range at a specific Ignite event
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(self.event, self.range_pop)

    def range_pop(self):
        _nvtx.rangePop()


class MarkHandler:
    """
    Mark an instantaneous event that occurred at some point.

    Args:
        msg: ASCII message to associate with range
    """

    def __init__(self, event: Events, msg: Optional[str] = None) -> None:
        if isinstance(event, str):
            event = event.upper()
        self.event = Events[event]
        if msg is None:
            msg = self.event.name
        self.msg = msg

    def attach(self, engine: Engine) -> None:
        """
        Add an NVTX mark to a specific Ignite event
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(self.event, self.mark)

    def mark(self):
        _nvtx.markA(self.msg)

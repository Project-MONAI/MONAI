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

from enum import Enum
from threading import RLock, Thread
from typing import TYPE_CHECKING, Callable, Dict

from monai.utils import exact_version, optional_import


if TYPE_CHECKING:
    from ignite.engine import Engine, Events
    import matplotlib.pyplot as plt

    has_matplotlib = True
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")
    Events, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Events")
    plt, has_matplotlib = optional_import("matplotlib.pyplot")


def _get_loss_from_output(output):
    return output["loss"].item()


class StatusMembers(Enum):
    """
    Named members of the status dictionary, others may be present for named metric values.
    """

    STATUS = "Status"
    EPOCHS = "Epochs"
    ITERS = "Iters"
    LOSS = "Loss"


class ThreadContainer(Thread):
    """
    Contains a running `Engine` object within a separate thread from main thread in a Jupyter notebook. This 
    allows an engine to begin a run in the background and allow the starting notebook cell to complete. A
    user can thus start a run and then navigate away from the notebook without concern for loosing connection
    with the running cell. All output is acquired through methods which synchronize with the running engine
    using an internal `lock` member, acquiring this lock allows the engine to be inspected while it's prevented
    from starting the next iteration.
    
    Args:
        engine: wrapped `Engine` object, when the container is started its `run` method is called
        loss_transform: callable to convert an output dict into a single numeric value
        metric_transform: callable to convert a metric value into a single numeric value
    """

    def __init__(
        self, engine: Engine, loss_transform: Callable = _get_loss_from_output, metric_transform: Callable = lambda x: x
    ):
        super().__init__()
        self.lock = RLock()
        self.engine = engine
        self._status_dict = {}
        self.loss_transform = loss_transform
        self.metric_transform = metric_transform

        self.engine.add_event_handler(Events.ITERATION_COMPLETED, self._update_status)

    def run(self):
        """Calls the `run` method of the wrapped engine."""
        self.engine.run()

    def stop(self):
        """Stop the engine and join the thread."""
        self.engine.terminate()
        self.join()

    def is_running(self) -> bool:
        """Returns True if the thread is still alive and the engine is still running."""
        return self.is_alive()

    def _update_status(self):
        """Called as an event, updates the internal status dict at the end of iterations."""
        with self.lock:
            state = self.engine.state

            if state is not None:
                if state.max_epochs > 1:
                    epoch = f"{state.epoch}/{state.max_epochs}"
                else:
                    epoch = str(state.epoch)

                if state.epoch_length is not None:
                    iters = f"{state.iteration % state.epoch_length}/{state.epoch_length}"
                else:
                    iters = str(state.iteration)

                stats[StatusMembers.EPOCHS.value] = epoch
                stats[StatusMembers.ITERS.value] = iters
                stats[StatusMembers.LOSS.value] = self.loss_transform(state.output)

                metrics = state.metrics or {}
                for m, v in metrics.items():
                    v = self.metric_transform(v)
                    if v is not None:
                        stats[m].append(v)

            self._status_dict.update(stats)

    @property
    def status_dict(self) -> Dict[str, str]:
        """A dictionary containing status information, current loss, and current metric values."""
        with self.lock:
            stats = {StatusMembers.STATUS.value: "Running" if self.is_running() else "Stopped"}
            stats.update(self._status_dict)
            return stats

    def status(self) -> str:
        """Returns a status string for the current state of the engine."""
        stats = self.status_dict

        msgs = [stats.pop(StatusMembers.STATUS.value), "Iters: " + str(stats.pop(StatusMembers.ITERS.value))]
        msgs += ["%s: %s" % kv for kv in stats.items()]

        return ", ".join(msgs)

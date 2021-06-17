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
This set of utility function is meant to make using Jupyter notebooks easier with MONAI. Plotting functions using
Matplotlib produce common plots for metrics and images.
"""

from enum import Enum
from threading import RLock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt

    has_matplotlib = True
except ImportError:
    has_matplotlib = False

try:
    from ignite.engine import Engine, Events

    has_ignite = True
except ImportError:
    Engine = object
    Events = object
    has_ignite = False

LOSS_NAME = "loss"


def plot_metric_graph(
    ax,
    title: str,
    graphmap: Dict[str, Union[List[float], Tuple[List[float], List[float]]]],
    yscale: str = "log",
    avg_keys: Tuple[str] = (LOSS_NAME,),
    window_fraction: int = 20,
):
    """
    Plot metrics on a single graph with running averages plotted for selected keys. The values in `graphmap`
    should be lists of (timepoint, value) pairs as stored in MetricLogger objects.

    Args:
        ax: Axes object to plot into
        title: graph title
        graphmap: dictionary of named graph values, which are lists of values or (index, value) pairs
        yscale: scale for y-axis compatible with `Axes.set_yscale`
        avg_keys: tuple of keys in `graphmap` to provide running average plots for
        window_fraction: what fraction of the graph value length to use as the running average window
    """
    from matplotlib.ticker import MaxNLocator

    for n, v in graphmap.items():
        if len(v) > 0:
            if isinstance(v[0], (tuple, list)):  # values are (x,y) pairs
                inds, vals = zip(*v)  # separate values into list of indices in X dimension and values
            else:
                inds, vals = tuple(range(len(v))), tuple(v)  # values are without indices, make indices for them

            ax.plot(inds, vals, label=f"{n} = {vals[-1]:.5g}")

            # if requested compute and plot a running average for the values using a fractional window size
            if n in avg_keys and len(v) > window_fraction:
                window = len(v) // window_fraction
                kernel = np.ones((window,)) / window
                ra = np.convolve((vals[0],) * (window - 1) + vals, kernel, mode="valid")

                ax.plot(inds, ra, label=f"{n} Avg = {ra[-1]:.5g}")

    ax.set_title(title)
    ax.set_yscale(yscale)
    ax.axis("on")
    ax.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.0)
    ax.grid(True, "both", "both")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_metric_images(
    fig,
    title: str,
    graphmap: Dict[str, Union[List[float], Tuple[List[float], List[float]]]],
    imagemap: Dict[str, np.ndarray],
    yscale: str = "log",
    avg_keys: Tuple[str] = (LOSS_NAME,),
    window_fraction: int = 20,
) -> List:
    """
    Plot metric graph data with images below into figure `fig`. The intended use is for the graph data to be
    metrics from a training run and the images to be the batch and output from the last iteration. This uses
    `plot_metric_graph` to plot the metric graph.

    Args:
        fig: Figure object to plot into, reuse from previous plotting for flicker-free refreshing
        title: graph title
        graphmap: dictionary of named graph values, which are lists of values or (index, value) pairs
        imagemap: dictionary of named images to show with metric plot
        yscale: for metric plot, scale for y-axis compatible with `Axes.set_yscale`
        avg_keys: for metric plot, tuple of keys in `graphmap` to provide running average plots for
        window_fraction: for metric plot, what fraction of the graph value length to use as the running average window

    Returns:
        list of Axes objects for graph followed by images
    """
    gridshape = (4, max(1, len(imagemap)))

    graph = plt.subplot2grid(gridshape, (0, 0), colspan=gridshape[1], fig=fig)

    plot_metric_graph(graph, title, graphmap, yscale, avg_keys, window_fraction)

    axes = [graph]
    for i, n in enumerate(imagemap):
        im = plt.subplot2grid(gridshape, (1, i), rowspan=2, fig=fig)

        if imagemap[n].shape[0] == 3:
            im.imshow(imagemap[n].transpose([1, 2, 0]))
        else:
            im.imshow(np.squeeze(imagemap[n]), cmap="gray")

        im.set_title("%s\n%.3g -> %.3g" % (n, imagemap[n].min(), imagemap[n].max()))
        im.axis("off")
        axes.append(im)

    return axes


def tensor_to_images(name: str, tensor: torch.Tensor):
    """
    Return an tuple of images derived from the given tensor. The `name` value indices which key from the
    output or batch value the tensor was stored as, or is "Batch" or "Output" if these were single tensors
    instead of dictionaries. Returns a tuple of 2D images of shape HW, or 3D images of shape CHW where C is
    color channels RGB or RGBA. This allows multiple images to be created from a single tensor, ie. to show
    each channel separately.
    """
    if tensor.ndim == 4 and tensor.shape[2] > 2 and tensor.shape[3] > 2:
        return tuple(tensor[0].cpu().data.numpy())
    if tensor.ndim == 5 and tensor.shape[3] > 2 and tensor.shape[4] > 2:
        dmid = tensor.shape[2] // 2
        return tuple(tensor[0, :, dmid].cpu().data.numpy())

    return ()


def plot_engine_status(
    engine: Engine,
    logger,
    title: str = "Training Log",
    yscale: str = "log",
    avg_keys: Tuple[str] = (LOSS_NAME,),
    window_fraction: int = 20,
    image_fn: Optional[Callable] = tensor_to_images,
    fig=None,
) -> Tuple:
    """
    Plot the status of the given Engine with its logger. The plot will consist of a graph of loss values and metrics
    taken from the logger, and images taken from the `output` and `batch` members of `engine.state`. The images are
    converted to Numpy arrays suitable for input to `Axes.imshow` using `image_fn`, if this is None then no image
    plotting is done.

    Args:
        engine: Engine to extract images from
        logger: MetricLogger to extract loss and metric data from
        title: graph title
        yscale: for metric plot, scale for y-axis compatible with `Axes.set_yscale`
        avg_keys: for metric plot, tuple of keys in `graphmap` to provide running average plots for
        window_fraction: for metric plot, what fraction of the graph value length to use as the running average window
        image_fn: callable converting tensors keyed to a name in the Engine to a tuple of images to plot
        fig: Figure object to plot into, reuse from previous plotting for flicker-free refreshing

    Returns:
        Figure object (or `fig` if given), list of Axes objects for graph and images
    """
    if fig is not None:
        fig.clf()
    else:
        fig = plt.Figure(figsize=(20, 10), tight_layout=True, facecolor="white")

    graphmap = {LOSS_NAME: logger.loss}
    graphmap.update(logger.metrics)

    imagemap = {}

    if image_fn is not None and engine.state is not None and engine.state.batch is not None:
        for src in (engine.state.batch, engine.state.output):
            if isinstance(src, dict):
                for k, v in src.items():
                    if isinstance(v, torch.Tensor):
                        images = image_fn(k, v)

                        for i, im in enumerate(images):
                            imagemap[f"{k}_{i}"] = im
            elif isinstance(src, torch.Tensor):
                label = "Batch" if src is engine.state.batch else "Output"
                images = image_fn(label, src)

                for i, im in enumerate(images):
                    imagemap[f"{label}_{i}"] = im

    axes = plot_metric_images(fig, title, graphmap, imagemap, yscale, avg_keys, window_fraction)

    if logger.loss:
        axes[0].axhline(logger.loss[-1][1], c="k", ls=":")  # draw dotted horizontal line at last loss value

    return fig, axes


def _get_loss_from_output(output: Union[Dict[str, torch.Tensor], torch.Tensor]) -> float:
    """Returns a single value from the network output, which is a dict or tensor."""
    if isinstance(output, dict):
        return output["loss"].item()
    return output.item()


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
        metric_transform: callable to convert a named metric value into a single numeric value
        status_format: format string for status key-value pairs.
    """

    def __init__(
        self,
        engine: Engine,
        loss_transform: Callable = _get_loss_from_output,
        metric_transform: Callable = lambda name, value: value,
        status_format: str = "{}: {:.4}",
    ):
        super().__init__()
        self.lock = RLock()
        self.engine = engine
        self._status_dict: Dict[str, Any] = {}
        self.loss_transform = loss_transform
        self.metric_transform = metric_transform
        self.fig = None
        self.status_format = status_format

        self.engine.add_event_handler(Events.ITERATION_COMPLETED, self._update_status)

    def run(self):
        """Calls the `run` method of the wrapped engine."""
        self.engine.run()

    def stop(self):
        """Stop the engine and join the thread."""
        self.engine.terminate()
        self.join()

    def _update_status(self):
        """Called as an event, updates the internal status dict at the end of iterations."""
        with self.lock:
            state = self.engine.state
            stats = {
                StatusMembers.EPOCHS.value: 0,
                StatusMembers.ITERS.value: 0,
                StatusMembers.LOSS.value: float("nan"),
            }

            if state is not None:
                if state.max_epochs >= 1:
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
                    v = self.metric_transform(m, v)
                    if v is not None:
                        stats[m].append(v)

            self._status_dict.update(stats)

    @property
    def status_dict(self) -> Dict[str, str]:
        """A dictionary containing status information, current loss, and current metric values."""
        with self.lock:
            stats = {StatusMembers.STATUS.value: "Running" if self.is_alive() else "Stopped"}
            stats.update(self._status_dict)
            return stats

    def status(self) -> str:
        """Returns a status string for the current state of the engine."""
        stats = self.status_dict

        msgs = [stats.pop(StatusMembers.STATUS.value), "Iters: " + str(stats.pop(StatusMembers.ITERS.value, 0))]

        for key, val in stats.items():
            if isinstance(val, float):
                msg = self.status_format.format(key, val)
            else:
                msg = f"{key}: {val}"

            msgs.append(msg)

        return ", ".join(msgs)

    def plot_status(self, logger, plot_func: Callable = plot_engine_status):
        """
        Generate a plot of the current status of the contained engine whose loss and metrics were tracked by `logger`.
        The function `plot_func` must accept arguments `title`, `engine`, `logger`, and `fig` which are the plot title,
        `self.engine`, `logger`, and `self.fig` respectively. The return value must be a figure object (stored in
        `self.fig`) and a list of Axes objects for the plots in the figure. Only the figure is returned by this method,
        which holds the internal lock during the plot generation.
        """
        with self.lock:
            self.fig, _ = plot_func(title=self.status(), engine=self.engine, logger=logger, fig=self.fig)
            return self.fig

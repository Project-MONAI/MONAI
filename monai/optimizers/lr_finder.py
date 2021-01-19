import copy
import os
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.core.arrayprint import _none_or_positive_arg
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from monai.networks.utils import eval_mode

try:
    import tqdm

    has_tqdm = True
except ImportError:
    has_tqdm = False
try:
    import matplotlib.pyplot as plt

    has_matplotlib = True
except ImportError:
    has_matplotlib = False

__all__ = ["LearningRateFinder"]


class DataLoaderIter(object):
    def __init__(self, data_loader: DataLoader, image_extractor: Callable, label_extractor: Callable) -> None:
        if not isinstance(data_loader, DataLoader):
            raise ValueError(
                f"Loader has unsupported type: {type(data_loader)}. Expected type was `torch.utils.data.DataLoader`"
            )
        self.data_loader = data_loader
        self._iterator = iter(data_loader)
        self.image_extractor = image_extractor
        self.label_extractor = label_extractor

    @property
    def dataset(self):
        return self.data_loader.dataset

    def inputs_labels_from_batch(self, batch_data):
        images = self.image_extractor(batch_data)
        labels = self.label_extractor(batch_data)
        return images, labels

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._iterator)
        return self.inputs_labels_from_batch(batch)


class TrainDataLoaderIter(DataLoaderIter):
    def __init__(
        self, data_loader: DataLoader, image_extractor: Callable, label_extractor: Callable, auto_reset: bool = True
    ) -> None:
        super().__init__(data_loader, image_extractor, label_extractor)
        self.auto_reset = auto_reset

    def __next__(self):
        try:
            batch = next(self._iterator)
            inputs, labels = self.inputs_labels_from_batch(batch)
        except StopIteration:
            if not self.auto_reset:
                raise
            self._iterator = iter(self.data_loader)
            batch = next(self._iterator)
            inputs, labels = self.inputs_labels_from_batch(batch)

        return inputs, labels


class ValDataLoaderIter(DataLoaderIter):
    """This iterator will reset itself **only** when it is acquired by
    the syntax of normal `iterator`. That is, this iterator just works
    like a `torch.data.DataLoader`. If you want to restart it, you
    should use it like:

        ```
        loader_iter = ValDataLoaderIter(data_loader)
        for batch in loader_iter:
            ...

        # `loader_iter` should run out of values now, you can restart it by:
        # 1. the way we use a `torch.data.DataLoader`
        for batch in loader_iter:        # __iter__ is called implicitly
            ...

        # 2. passing it into `iter()` manually
        loader_iter = iter(loader_iter)  # __iter__ is called by `iter()`
        ```
    """

    def __init__(self, data_loader: DataLoader, image_extractor: Callable, label_extractor: Callable) -> None:
        super().__init__(data_loader, image_extractor, label_extractor)
        self.run_limit = len(self.data_loader)
        self.run_counter = 0

    def __iter__(self):
        if self.run_counter >= self.run_limit:
            self._iterator = iter(self.data_loader)
            self.run_counter = 0
        return self

    def __next__(self):
        self.run_counter += 1
        return super(ValDataLoaderIter, self).__next__()


def default_image_extractor(x: Any) -> torch.Tensor:
    """Default callable for getting image from batch data."""
    out: torch.Tensor = x["image"] if isinstance(x, dict) else x[0]
    return out


def default_label_extractor(x: Any) -> torch.Tensor:
    """Default callable for getting label from batch data."""
    out: torch.Tensor = x["label"] if isinstance(x, dict) else x[1]
    return out


class LearningRateFinder(object):
    """Learning rate range test.

    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.

    Example (fastai approach):
        >>> lr_finder = LearningRateFinder(net, optimizer, criterion)
        >>> lr_finder.range_test(data_loader, end_lr=100, num_iter=100)
        >>> lr_finder.get_steepest_gradient()
        >>> lr_finder.plot() # to inspect the loss-learning rate graph

    Example (Leslie Smith's approach):
        >>> lr_finder = LearningRateFinder(net, optimizer, criterion)
        >>> lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=1, num_iter=100, step_mode="linear")

    Gradient accumulation is supported; example:
        >>> train_data = ...    # prepared dataset
        >>> desired_bs, real_bs = 32, 4         # batch size
        >>> accumulation_steps = desired_bs // real_bs     # required steps for accumulation
        >>> data_loader = torch.utils.data.DataLoader(train_data, batch_size=real_bs, shuffle=True)
        >>> acc_lr_finder = LearningRateFinder(net, optimizer, criterion)
        >>> acc_lr_finder.range_test(data_loader, end_lr=10, num_iter=100, accumulation_steps=accumulation_steps)

    By default, image will be extracted from data loader with x["image"] and x[0], depending on whether
    batch data is a dictionary or not (and similar behaviour for extracting the label). If your data loader
    returns something other than this, pass a callable function to extract it, e.g.:
        >>> image_extractor = lambda x: x["input"]
        >>> label_extractor = lambda x: x[100]
        >>> lr_finder = LearningRateFinder(net, optimizer, criterion)
        >>> lr_finder.range_test(train_loader, val_loader, image_extractor, label_extractor)

    References:
    Modified from: https://github.com/davidtvs/pytorch-lr-finder.
    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: torch.nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        memory_cache: bool = True,
        cache_dir: Optional[str] = None,
        amp: bool = False,
        verbose: bool = True,
    ) -> None:
        """Constructor.

        Args:
            model: wrapped model.
            optimizer: wrapped optimizer.
            criterion: wrapped loss function.
            device: device on which to test. run a string ("cpu" or "cuda") with an
                optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
                Alternatively, can be an object representing the device on which the
                computation will take place. Default: None, uses the same device as `model`.
            memory_cache: if this flag is set to True, `state_dict` of
                model and optimizer will be cached in memory. Otherwise, they will be saved
                to files under the `cache_dir`.
            cache_dir: path for storing temporary files. If no path is
                specified, system-wide temporary directory is used. Notice that this
                parameter will be ignored if `memory_cache` is True.
            amp: use Automatic Mixed Precision
            verbose: verbose output
        Returns:
            None
        """
        # Check if the optimizer is already attached to a scheduler
        self.optimizer = optimizer
        self._check_for_scheduler()

        self.model = model
        self.criterion = criterion
        self.history: Dict[str, list] = {"lr": [], "loss": []}
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir
        self.amp = amp
        self.verbose = verbose

        # Save the original state of the model and optimizer so they can be restored if
        # needed
        self.model_device = next(self.model.parameters()).device
        self.state_cacher = StateCacher(memory_cache, cache_dir=cache_dir)
        self.state_cacher.store("model", self.model.state_dict())
        self.state_cacher.store("optimizer", self.optimizer.state_dict())

        # If device is None, use the same as the model
        self.device = device if device else self.model_device

    def reset(self) -> None:
        """Restores the model and optimizer to their initial states."""

        self.model.load_state_dict(self.state_cacher.retrieve("model"))
        self.optimizer.load_state_dict(self.state_cacher.retrieve("optimizer"))
        self.model.to(self.model_device)

    def range_test(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        image_extractor: Callable = default_image_extractor,
        label_extractor: Callable = default_label_extractor,
        start_lr: Optional[float] = None,
        end_lr: int = 10,
        num_iter: int = 100,
        step_mode: str = "exp",
        smooth_f: float = 0.05,
        diverge_th: int = 5,
        accumulation_steps: int = 1,
        non_blocking_transfer: bool = True,
        auto_reset: bool = True,
    ) -> None:
        """Performs the learning rate range test.

        Args:
            train_loader: training set data loader.
            val_loader: validation data loader (if desired).
            image_extractor: callable function to get the image from a batch of data.
                Default: `x["image"] if isinstance(x, dict) else x[0]`.
            label_extractor: callable function to get the label from a batch of data.
                Default: `x["label"] if isinstance(x, dict) else x[1]`.
            start_lr : the starting learning rate for the range test.
                The default is the optimizer's learning rate.
            end_lr: the maximum learning rate to test. The test may stop earlier than
                this if the result starts diverging.
            num_iter: the max number of iterations for test.
            step_mode: schedule for increasing learning rate: (`linear` or `exp`).
            smooth_f: the loss smoothing factor within the `[0, 1[` interval. Disabled
                if set to `0`, otherwise loss is smoothed using exponential smoothing.
            diverge_th: test is stopped when loss surpasses threshold:
                `diverge_th * best_loss`.
            accumulation_steps: steps for gradient accumulation. If set to `1`,
                gradients are not accumulated.
            non_blocking_transfer: when `True`, moves data to device asynchronously if
                possible, e.g., moving CPU Tensors with pinned memory to CUDA devices.
            auto_reset: if `True`, returns model and optimizer to original states at end
                of test.
        Returns:
            None
        """

        # Reset test results
        self.history = {"lr": [], "loss": []}
        best_loss = -float("inf")

        # Move the model to the proper device
        self.model.to(self.device)

        # Check if the optimizer is already attached to a scheduler
        self._check_for_scheduler()

        # Set the starting learning rate
        if start_lr:
            self._set_learning_rate(start_lr)

        # Check number of iterations
        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")

        # Initialize the proper learning rate policy
        lr_schedule: Union[ExponentialLR, LinearLR]
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError(f"expected one of (exp, linear), got {step_mode}")

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1[")

        # Create an iterator to get data batch by batch
        train_iter = TrainDataLoaderIter(train_loader, image_extractor, label_extractor)
        if val_loader:
            val_iter = ValDataLoaderIter(val_loader, image_extractor, label_extractor)

        trange: Union[partial[tqdm.trange], Type[range]]
        if self.verbose and has_tqdm:
            trange = partial(tqdm.trange, desc="Computing optimal learning rate")
            tprint = tqdm.tqdm.write
        else:
            trange = range
            tprint = print

        for iteration in trange(num_iter):
            if self.verbose and not has_tqdm:
                print(f"Computing optimal learning rate, iteration {iteration + 1}/{num_iter}")

            # Train on batch and retrieve loss
            loss = self._train_batch(
                train_iter,
                accumulation_steps,
                non_blocking_transfer=non_blocking_transfer,
            )
            if val_loader:
                loss = self._validate(val_iter, non_blocking_transfer=non_blocking_transfer)

            # Update the learning rate
            self.history["lr"].append(lr_schedule.get_lr()[0])
            lr_schedule.step()

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
                if loss < best_loss:
                    best_loss = loss

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            if loss > diverge_th * best_loss:
                if self.verbose:
                    tprint("Stopping early, the loss has diverged")
                break

        if auto_reset:
            if self.verbose:
                print("Resetting model and optimizer")
            self.reset()

    def _set_learning_rate(self, new_lrs: Union[float, list]) -> None:
        """Set learning rate(s) for optimizer."""
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Length of `new_lrs` is not equal to the number of parameter groups " + "in the given optimizer"
            )

        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr

    def _check_for_scheduler(self) -> _none_or_positive_arg:
        """Check optimizer doesn't already have scheduler."""
        for param_group in self.optimizer.param_groups:
            if "initial_lr" in param_group:
                raise RuntimeError("Optimizer already has a scheduler attached to it")

    def _train_batch(self, train_iter, accumulation_steps: int, non_blocking_transfer: bool = True) -> float:
        self.model.train()
        total_loss = 0

        self.optimizer.zero_grad()
        for i in range(accumulation_steps):
            inputs, labels = next(train_iter)
            inputs, labels = self._move_to_device(inputs, labels, non_blocking=non_blocking_transfer)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Loss should be averaged in each step
            loss /= accumulation_steps

            # Backward pass
            if self.amp and hasattr(self.optimizer, "_amp_stash"):
                # For minor performance optimization, see also:
                # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
                delay_unscale = ((i + 1) % accumulation_steps) != 0

                with torch.cuda.amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:  # type: ignore
                    scaled_loss.backward()
            else:
                loss.backward()

            total_loss += loss.item()

        self.optimizer.step()

        return total_loss

    def _move_to_device(
        self, inputs: torch.Tensor, labels: torch.Tensor, non_blocking: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def move(obj, device, non_blocking=True):
            if hasattr(obj, "to"):
                return obj.to(device, non_blocking=non_blocking)
            elif isinstance(obj, tuple):
                return tuple(move(o, device, non_blocking) for o in obj)
            elif isinstance(obj, list):
                return [move(o, device, non_blocking) for o in obj]
            elif isinstance(obj, dict):
                return {k: move(o, device, non_blocking) for k, o in obj.items()}
            else:
                return obj

        inputs = move(inputs, self.device, non_blocking=non_blocking)
        labels = move(labels, self.device, non_blocking=non_blocking)
        return inputs, labels

    def _validate(self, val_iter: ValDataLoaderIter, non_blocking_transfer: bool = True) -> float:
        # Set model to evaluation mode and disable gradient computation
        running_loss = 0
        with eval_mode(self.model):
            for inputs, labels in val_iter:
                # Move data to the correct device
                inputs, labels = self._move_to_device(inputs, labels, non_blocking=non_blocking_transfer)

                # Forward pass and loss computation
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * len(labels)

        return running_loss / len(val_iter.dataset)

    def get_steepest_gradient(
        self,
        skip_start: int = 10,
        skip_end: int = 5,
    ) -> Union[Tuple[float, float], Tuple[None, None]]:
        """Get learning rate which has steepest gradient and its corresponding loss

        Args:
            skip_start: number of batches to trim from the start.
            skip_end: number of batches to trim from the end.

        Returns:
            Learning rate which has steepest gradient and its corresponding loss
        """
        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        try:
            min_grad_idx = np.gradient(np.array(losses)).argmin()
            return lrs[min_grad_idx], losses[min_grad_idx]
        except ValueError:
            print("Failed to compute the gradients, there might not be enough points.")
            return None, None

    def plot(
        self,
        skip_start: int = 10,
        skip_end: int = 5,
        log_lr: bool = True,
        ax=None,
        steepest_lr: bool = True,
    ):
        """Plots the learning rate range test.

        Args:
            skip_start: number of batches to trim from the start.
            skip_end: number of batches to trim from the start.
            log_lr: True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale.
            ax: the plot is created in the specified matplotlib axes object and the
                figure is not be shown. If `None`, then the figure and axes object are
                created in this method and the figure is shown.
            steepest_lr: plot the learning rate which had the steepest gradient.

        Returns:
            The matplotlib.axes.Axes object that contains the plot
        """
        if not has_matplotlib:
            raise RuntimeError("Matplotlib is missing, can't plot result")

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Create the figure and axes object if axes was not already given
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)

        # Plot the LR with steepest gradient
        if steepest_lr:
            lr_at_steepest_grad, loss_at_steepest_grad = self.get_steepest_gradient(skip_start, skip_end)
            if lr_at_steepest_grad is not None:
                ax.scatter(
                    lr_at_steepest_grad,
                    loss_at_steepest_grad,
                    s=75,
                    marker="o",
                    color="red",
                    zorder=3,
                    label="steepest gradient",
                )
                ax.legend()

        if log_lr:
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")

        # Show only if the figure was created internally
        if fig is not None:
            plt.show()

        return ax


class _LRSchedulerMONAI(_LRScheduler):
    def __init__(self, optimizer: Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            end_lr: the final learning rate.
            num_iter: the number of iterations over which the test occurs.
            last_epoch: the index of last epoch.
        Returns:
            None
        """
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_LRSchedulerMONAI, self).__init__(optimizer, last_epoch)


class LinearLR(_LRSchedulerMONAI):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    """

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRSchedulerMONAI):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    """

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class StateCacher(object):
    def __init__(self, in_memory: bool, cache_dir: Optional[str] = None) -> None:
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile

            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError("Given `cache_dir` is not a valid directory.")

        self.cached: Dict[str, str] = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, f"state_{key}_{id(self)}.pt")
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError(f"Target {key} was not cached.")

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)  # pytype: disable=attribute-error
            if not os.path.exists(fn):  # pytype: disable=wrong-arg-types
                raise RuntimeError(f"Failed to load state in {fn}. File doesn't exist anymore.")
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """If necessary, delete any cached files existing in `cache_dir`."""
        if self.in_memory:
            return
        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])

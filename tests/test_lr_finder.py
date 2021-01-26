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

from functools import partial
import os
import random
import sys
import unittest

import torch
from torch.utils.data import DataLoader

from monai.apps import MedNISTDataset
from monai.networks.nets import DenseNet
from monai.networks.utils import eval_mode
from monai.optimizers import LearningRateFinder
from monai.transforms import AddChanneld, Compose, LoadImaged, ScaleIntensityd, ToTensord
from monai.utils import copy_to_device, optional_import, set_determinism

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, Union

PILImage, has_pil = optional_import("PIL.Image")

RAND_SEED = 42
random.seed(RAND_SEED)
set_determinism(seed=RAND_SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


class DataLoaderIter:
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


def train_valid_loss(
    train_iter,
    val_iter,
    model,
    optimizer,
    criterion,
    accumulation_steps,
    device,
    non_blocking_transfer,
    amp,
):
    model.train()
    total_loss = 0

    optimizer.zero_grad()
    for i in range(accumulation_steps):
        inputs, labels = next(train_iter)
        inputs, labels = copy_to_device([inputs, labels], device=device, non_blocking=non_blocking_transfer)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Loss should be averaged in each step
        loss /= accumulation_steps

        # Backward pass
        if amp and hasattr(optimizer, "_amp_stash"):
            # For minor performance optimization, see also:
            # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
            delay_unscale = ((i + 1) % accumulation_steps) != 0

            with torch.cuda.amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale) as scaled_loss:  # type: ignore
                scaled_loss.backward()
        else:
            loss.backward()

        total_loss += loss.item()

    optimizer.step()

    if not val_iter:
        return total_loss

    # Set model to evaluation mode and disable gradient computation
    running_loss = 0
    with eval_mode(model):
        for inputs, labels in val_iter:
            # Copy data to the correct device
            inputs, labels = copy_to_device(
                [inputs, labels], device=device, non_blocking=non_blocking_transfer
            )

            # Forward pass and loss computation
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * len(labels)

    return running_loss / len(val_iter.dataset)


@unittest.skipUnless(sys.platform == "linux", "requires linux")
@unittest.skipUnless(has_pil, "requires PIL")
class TestLRFinder(unittest.TestCase):
    def setUp(self):

        self.root_dir = os.environ.get("MONAI_DATA_DIRECTORY")
        if not self.root_dir:
            self.root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")

        self.transforms = Compose(
            [
                LoadImaged(keys="image"),
                AddChanneld(keys="image"),
                ScaleIntensityd(keys="image"),
                ToTensord(keys="image"),
            ]
        )

    def test_lr_finder(self):
        # 0.001 gives 54 examples
        train_ds = MedNISTDataset(
            root_dir=self.root_dir,
            transform=self.transforms,
            section="validation",
            val_frac=0.001,
            download=True,
            num_workers=10,
        )
        train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)
        num_classes = train_ds.get_num_classes()

        model = DenseNet(
            spatial_dims=2, in_channels=1, out_channels=num_classes, init_features=2, growth_rate=2, block_config=(2,)
        )
        loss_function = torch.nn.CrossEntropyLoss()
        learning_rate = 1e-5
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        train_iter = TrainDataLoaderIter(train_loader, default_image_extractor, default_label_extractor)
        val_iter = ValDataLoaderIter(train_loader, default_image_extractor, default_label_extractor)

        train_valid_loss_iter = partial(
            train_valid_loss,
            train_iter,
            val_iter,
            model = model,
            optimizer = optimizer,
            criterion = loss_function,
            accumulation_steps = 1,
            device = device,
            non_blocking_transfer = True,
            amp = False,
        )

        lr_finder = LearningRateFinder(model, optimizer, device=device)
        lr_finder.range_test(train_valid_loss_iter=train_valid_loss_iter, end_lr=10, num_iter=5)
        print(lr_finder.get_steepest_gradient(0, 0)[0])
        lr_finder.plot(0, 0)  # to inspect the loss-learning rate graph
        lr_finder.reset()  # to reset the model and optimizer to their initial state


if __name__ == "__main__":
    # unittest.main()
    a = TestLRFinder()
    a.setUp()
    a.test_lr_finder()

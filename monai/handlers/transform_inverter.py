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

import warnings
from typing import TYPE_CHECKING, Callable, Optional

from torch.utils.data import DataLoader as TorchDataLoader

from monai.data import BatchInverseTransform
from monai.data.utils import no_collation
from monai.engines.utils import CommonKeys
from monai.transforms import InvertibleTransform, allow_missing_keys_mode
from monai.utils import InverseKeys, exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")


class TransformInverter:
    """
    Ignite handler to automatically invert all the pre-transforms that support `inverse`.
    It takes `engine.state.output` as the input data and uses the transforms infomation from `engine.state.batch`.

    Note:
        This handler is experimental API in v0.5, the interpolation mode in the transforms
        and inverse transforms are the same, so maybe it's not correct as we may want to use `bilinear`
        for input image but use `nearest` when inverting transforms for model outout.
        For this case, a solution is to set `batch_key` to the label field if we have labels.

    """

    def __init__(
        self,
        transform: InvertibleTransform,
        loader: TorchDataLoader,
        collate_fn: Optional[Callable] = no_collation,
        batch_key: str = CommonKeys.IMAGE,
        output_key: str = CommonKeys.PRED,
        postfix: str = "inverted",
    ) -> None:
        """
        Args:
            transform: a callable data transform on input data.
            loader: data loader used to generate the batch of data.
            collate_fn: how to collate data after inverse transformations.
                default won't do any collation, so the output will be a list of size batch size.
            batch_key: the key of input data in `ignite.engine.batch`. will get the applied transforms
                for this input data, then invert them for the model output, default to "image".
            output_key: the key of model output in `ignite.engine.output`, invert transforms on it.
            postfix: will save the inverted result into `ignite.engine.output` with key `{ouput_key}_{postfix}`.

        """
        self.transform = transform
        self.inverter = BatchInverseTransform(transform=transform, loader=loader, collate_fn=collate_fn)
        self.batch_key = batch_key
        self.output_key = output_key
        self.postfix = postfix

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        transform_key = self.batch_key + InverseKeys.KEY_SUFFIX
        if transform_key not in engine.state.batch:
            warnings.warn("all the pre-transforms are not InvertibleTransform or no need to invert.")
            return

        segs_dict = {
            self.batch_key: engine.state.output[self.output_key].detach().cpu(),
            transform_key: engine.state.batch[transform_key],
        }

        with allow_missing_keys_mode(self.transform):  # type: ignore
            inverted_key = f"{self.output_key}_{self.postfix}"
            engine.state.output[inverted_key] = [i[self.batch_key] for i in self.inverter(segs_dict)]

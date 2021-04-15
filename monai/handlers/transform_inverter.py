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
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

from torch.utils.data import DataLoader as TorchDataLoader

from monai.data import BatchInverseTransform
from monai.data.utils import no_collation
from monai.engines.utils import CommonKeys, IterationEvents
from monai.transforms import InvertibleTransform, ToTensor, allow_missing_keys_mode, convert_inverse_interp_mode
from monai.utils import InverseKeys, ensure_tuple, ensure_tuple_rep, exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")


class TransformInverter:
    """
    Ignite handler to automatically invert `transforms`.
    It takes `engine.state.output` as the input data and uses the transforms information from `engine.state.batch`.
    The outputs are stored in `engine.state.output` with key: "{output_key}_{postfix}".
    """

    def __init__(
        self,
        transform: InvertibleTransform,
        loader: TorchDataLoader,
        output_keys: Union[str, Sequence[str]] = CommonKeys.PRED,
        batch_keys: Union[str, Sequence[str]] = CommonKeys.IMAGE,
        meta_key_postfix: str = "meta_dict",
        collate_fn: Optional[Callable] = no_collation,
        postfix: str = "inverted",
        nearest_interp: Union[bool, Sequence[bool]] = True,
        num_workers: Optional[int] = 0,
    ) -> None:
        """
        Args:
            transform: a callable data transform on input data.
            loader: data loader used to run transforms and generate the batch of data.
            output_keys: the key of expected data in `ignite.engine.output`, invert transforms on it.
                it also can be a list of keys, will invert transform for each of them. Default to "pred".
            batch_keys: the key of input data in `ignite.engine.batch`. will get the applied transforms
                for this input data, then invert them for the expected data with `output_keys`.
                It can also be a list of keys, each matches to the `output_keys` data. default to "image".
            meta_key_postfix: use `{batch_key}_{postfix}` to to fetch the meta data according to the key data,
                default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle key `image`,  read/write affine matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.
            collate_fn: how to collate data after inverse transformations.
                default won't do any collation, so the output will be a list of size batch size.
            postfix: will save the inverted result into `ignite.engine.output` with key `{output_key}_{postfix}`.
            nearest_interp: whether to use `nearest` interpolation mode when inverting the spatial transforms,
                default to `True`. If `False`, use the same interpolation mode as the original transform.
                it also can be a list of bool, each matches to the `output_keys` data.
            num_workers: number of workers when run data loader for inverse transforms,
                default to 0 as only run one iteration and multi-processing may be even slower.
                Set to `None`, to use the `num_workers` of the input transform data loader.

        """
        self.transform = transform
        self.inverter = BatchInverseTransform(
            transform=transform,
            loader=loader,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        self.output_keys = ensure_tuple(output_keys)
        self.batch_keys = ensure_tuple_rep(batch_keys, len(self.output_keys))
        self.meta_key_postfix = meta_key_postfix
        self.postfix = postfix
        self.nearest_interp = ensure_tuple_rep(nearest_interp, len(self.output_keys))
        self._totensor = ToTensor()

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(IterationEvents.MODEL_COMPLETED, self)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        for output_key, batch_key, nearest_interp in zip(self.output_keys, self.batch_keys, self.nearest_interp):
            transform_key = batch_key + InverseKeys.KEY_SUFFIX
            if transform_key not in engine.state.batch:
                warnings.warn(f"all the transforms on `{batch_key}` are not InvertibleTransform.")
                continue

            transform_info = engine.state.batch[transform_key]
            if nearest_interp:
                transform_info = convert_inverse_interp_mode(
                    trans_info=deepcopy(transform_info),
                    mode="nearest",
                    align_corners=None,
                )

            segs_dict = {
                batch_key: engine.state.output[output_key].detach().cpu(),
                transform_key: transform_info,
            }
            meta_dict_key = f"{batch_key}_{self.meta_key_postfix}"
            if meta_dict_key in engine.state.batch:
                segs_dict[meta_dict_key] = engine.state.batch[meta_dict_key]

            with allow_missing_keys_mode(self.transform):  # type: ignore
                inverted_key = f"{output_key}_{self.postfix}"
                engine.state.output[inverted_key] = [self._totensor(i[batch_key]) for i in self.inverter(segs_dict)]

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
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

import torch

from monai.config import IgniteInfo, KeysCollection
from monai.engines.utils import CommonKeys, IterationEvents
from monai.transforms import Invertd, InvertibleTransform
from monai.utils import deprecated, ensure_tuple, ensure_tuple_rep, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


@deprecated(since="0.6.0", removed="0.7.0", msg_suffix="Please consider using `Invertd` transform instead.")
class TransformInverter:
    """
    Ignite handler to automatically invert `transforms`.
    It takes `engine.state.output` as the input data and uses the transforms information from `engine.state.batch`.
    Expect both `engine.state.output` and `engine.state.batch` to be list of dictionaries data.
    The inverted data is in-place saved back to `engine.state.output` with key: "{output_key}".
    And the inverted meta dict will be stored in `engine.state.batch`
    with key: "{meta_keys}" or "{key}_{meta_key_postfix}".

    """

    def __init__(
        self,
        transform: InvertibleTransform,
        output_keys: KeysCollection = CommonKeys.PRED,
        batch_keys: KeysCollection = CommonKeys.IMAGE,
        meta_keys: Optional[KeysCollection] = None,
        batch_meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        nearest_interp: Union[bool, Sequence[bool]] = True,
        to_tensor: Union[bool, Sequence[bool]] = True,
        device: Union[Union[str, torch.device], Sequence[Union[str, torch.device]]] = "cpu",
        post_func: Union[Callable, Sequence[Callable]] = lambda x: x,
        num_workers: Optional[int] = 0,
    ) -> None:
        """
        Args:
            transform: a callable data transform on input data.
            output_keys: the key of expected data in `ignite.engine.output`, invert transforms on it.
                it also can be a list of keys, will invert transform for each of them.
                Default to "pred". it's in-place operation.
            batch_keys: the key of input data in `ignite.engine.batch`. will get the applied transforms
                for this input data, then invert them for the expected data with `output_keys`.
                It can also be a list of keys, each matches to the `output_keys` data. default to "image".
            meta_keys: explicitly indicate the key for the inverted meta data dictionary.
                the meta data is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `{key}_{meta_key_postfix}`.
            batch_meta_keys: the key of the meta data of input data in `ignite.engine.batch`,
                will get the `affine`, `data_shape`, etc.
                the meta data is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `{orig_key}_{meta_key_postfix}`.
                meta data will also be inverted and stored in `meta_keys`.
            meta_key_postfix: if `orig_meta_keys` is None, use `{orig_key}_{meta_key_postfix}` to to fetch the
                meta data from dict, if `meta_keys` is None, use `{key}_{meta_key_postfix}`.
                default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle orig_key `image`,  read/write `affine` matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.
                the inverted meta dict will be stored with key: "{key}_{meta_key_postfix}".
            nearest_interp: whether to use `nearest` interpolation mode when inverting the spatial transforms,
                default to `True`. If `False`, use the same interpolation mode as the original transform.
                it also can be a list of bool, each matches to the `output_keys` data.
            to_tensor: whether to convert the inverted data into PyTorch Tensor first, default to `True`.
                it also can be a list of bool, each matches to the `output_keys` data.
            device: if converted to Tensor, move the inverted results to target device before `post_func`,
                default to "cpu", it also can be a list of string or `torch.device`,
                each matches to the `output_keys` data.
            post_func: post processing for the inverted data, should be a callable function.
                it also can be a list of callable, each matches to the `output_keys` data.

        """
        self.inverter = Invertd(
            keys=output_keys,
            transform=transform,
            orig_keys=batch_keys,
            meta_keys=meta_keys,
            orig_meta_keys=batch_meta_keys,
            meta_key_postfix=meta_key_postfix,
            nearest_interp=nearest_interp,
            to_tensor=to_tensor,
            device=device,
            post_func=post_func,
        )
        self.output_keys = ensure_tuple(output_keys)
        self.meta_keys = ensure_tuple_rep(None, len(self.output_keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.output_keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as output_keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.output_keys))

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
        if not isinstance(engine.state.batch, list) or not isinstance(engine.state.output, list):
            warnings.warn("inverter requires `engine.state.batch` and `engine.state.output` to be lists.")
        else:
            for i, (b, o) in enumerate(zip(engine.state.batch, engine.state.output)):
                # combine `batch` and `output` to temporarily act as 1 dict for postprocessing
                data = dict(b)
                data.update(o)
                ret = self.inverter(data)

                for output_key, meta_key, meta_key_postfix in zip(
                    self.output_keys, self.meta_keys, self.meta_key_postfix
                ):
                    # save the inverted data into state.output
                    engine.state.output[i][output_key] = ret.get(output_key)
                    # save the inverted meta dict into state.batch
                    meta_key = meta_key or f"{output_key}_{meta_key_postfix}"
                    if meta_key in ret:
                        # FIXME: we save inverted meta dict into `batch` to be compatible with `SegmentationSaver`
                        # will deprecate both handlers soon
                        engine.state.batch[i][meta_key] = ret.get(meta_key)

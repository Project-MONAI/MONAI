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
A collection of dictionary-based wrappers around the "vanilla" transforms for model output tensors
defined in :py:class:`monai.transforms.utility.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

import warnings
from copy import deepcopy
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

import monai.data
from monai.config import KeysCollection
from monai.data.utils import no_collation
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.inverse_batch_transform import BatchInverseTransform
from monai.transforms.post.array import (
    Activations,
    AsDiscrete,
    KeepLargestConnectedComponent,
    LabelToContour,
    MeanEnsemble,
    ProbNMS,
    VoteEnsemble,
)
from monai.transforms.transform import MapTransform
from monai.transforms.utility.array import ToTensor
from monai.transforms.utils import allow_missing_keys_mode, convert_inverse_interp_mode
from monai.utils import ensure_tuple_rep
from monai.utils.enums import InverseKeys

__all__ = [
    "Activationsd",
    "AsDiscreted",
    "KeepLargestConnectedComponentd",
    "LabelToContourd",
    "Ensembled",
    "MeanEnsembled",
    "VoteEnsembled",
    "ActivationsD",
    "ActivationsDict",
    "AsDiscreteD",
    "AsDiscreteDict",
    "InvertD",
    "InvertDict",
    "Invertd",
    "KeepLargestConnectedComponentD",
    "KeepLargestConnectedComponentDict",
    "LabelToContourD",
    "LabelToContourDict",
    "MeanEnsembleD",
    "MeanEnsembleDict",
    "VoteEnsembleD",
    "VoteEnsembleDict",
    "DecollateD",
    "DecollateDict",
    "Decollated",
    "ProbNMSd",
    "ProbNMSD",
    "ProbNMSDict",
]


class Activationsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddActivations`.
    Add activation layers to the input data specified by `keys`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigmoid: Union[Sequence[bool], bool] = False,
        softmax: Union[Sequence[bool], bool] = False,
        other: Optional[Union[Sequence[Callable], Callable]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sigmoid: whether to execute sigmoid function on model output before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            softmax: whether to execute softmax function on model output before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            other: callable function to execute other activation layers,
                for example: `other = torch.tanh`. it also can be a sequence of Callable, each
                element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.sigmoid = ensure_tuple_rep(sigmoid, len(self.keys))
        self.softmax = ensure_tuple_rep(softmax, len(self.keys))
        self.other = ensure_tuple_rep(other, len(self.keys))
        self.converter = Activations()

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key, sigmoid, softmax, other in self.key_iterator(d, self.sigmoid, self.softmax, self.other):
            d[key] = self.converter(d[key], sigmoid, softmax, other)
        return d


class AsDiscreted(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsDiscrete`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        argmax: Union[Sequence[bool], bool] = False,
        to_onehot: Union[Sequence[bool], bool] = False,
        n_classes: Optional[Union[Sequence[int], int]] = None,
        threshold_values: Union[Sequence[bool], bool] = False,
        logit_thresh: Union[Sequence[float], float] = 0.5,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            argmax: whether to execute argmax function on input data before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            to_onehot: whether to convert input data into the one-hot format. Defaults to False.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            n_classes: the number of classes to convert to One-Hot format. it also can be a
                sequence of int, each element corresponds to a key in ``keys``.
            threshold_values: whether threshold the float value to int number 0 or 1, default is False.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            logit_thresh: the threshold value for thresholding operation, default is 0.5.
                it also can be a sequence of float, each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.argmax = ensure_tuple_rep(argmax, len(self.keys))
        self.to_onehot = ensure_tuple_rep(to_onehot, len(self.keys))
        self.n_classes = ensure_tuple_rep(n_classes, len(self.keys))
        self.threshold_values = ensure_tuple_rep(threshold_values, len(self.keys))
        self.logit_thresh = ensure_tuple_rep(logit_thresh, len(self.keys))
        self.converter = AsDiscrete()

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key, argmax, to_onehot, n_classes, threshold_values, logit_thresh in self.key_iterator(
            d, self.argmax, self.to_onehot, self.n_classes, self.threshold_values, self.logit_thresh
        ):
            d[key] = self.converter(
                d[key],
                argmax,
                to_onehot,
                n_classes,
                threshold_values,
                logit_thresh,
            )
        return d


class KeepLargestConnectedComponentd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.KeepLargestConnectedComponent`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        applied_labels: Union[Sequence[int], int],
        independent: bool = True,
        connectivity: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            applied_labels: Labels for applying the connected component on.
                If only one channel. The pixel whose value is not in this list will remain unchanged.
                If the data is in one-hot format, this is the channel indices to apply transform.
            independent: consider several labels as a whole or independent, default is `True`.
                Example use case would be segment label 1 is liver and label 2 is liver tumor, in that case
                you want this "independent" to be specified as False.
            connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. If ``None``, a full
                connectivity of ``input.ndim`` is used.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.converter = KeepLargestConnectedComponent(applied_labels, independent, connectivity)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class LabelToContourd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LabelToContour`.
    """

    def __init__(self, keys: KeysCollection, kernel_type: str = "Laplace", allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            kernel_type: the method applied to do edge detection, default is "Laplace".
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.converter = LabelToContour(kernel_type=kernel_type)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class Ensembled(MapTransform):
    """
    Base class of dictionary-based ensemble transforms.

    """

    def __init__(
        self,
        keys: KeysCollection,
        ensemble: Callable[[Union[Sequence[torch.Tensor], torch.Tensor]], torch.Tensor],
        output_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
            ensemble: callable method to execute ensemble on specified data.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.
            allow_missing_keys: don't raise exception if key is missing.

        Raises:
            TypeError: When ``ensemble`` is not ``callable``.
            ValueError: When ``len(keys) > 1`` and ``output_key=None``. Incompatible values.

        """
        super().__init__(keys, allow_missing_keys)
        if not callable(ensemble):
            raise TypeError(f"ensemble must be callable but is {type(ensemble).__name__}.")
        self.ensemble = ensemble
        if len(self.keys) > 1 and output_key is None:
            raise ValueError("Incompatible values: len(self.keys) > 1 and output_key=None.")
        self.output_key = output_key if output_key is not None else self.keys[0]

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        items: Union[List[torch.Tensor], torch.Tensor]
        if len(self.keys) == 1:
            items = d[self.keys[0]]
        else:
            items = [d[key] for key in self.key_iterator(d)]
        d[self.output_key] = self.ensemble(items)

        return d


class MeanEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MeanEnsemble`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        output_key: Optional[str] = None,
        weights: Optional[Union[Sequence[float], torch.Tensor, np.ndarray]] = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.
            weights: can be a list or tuple of numbers for input data with shape: [E, B, C, H, W[, D]].
                or a Numpy ndarray or a PyTorch Tensor data.
                the `weights` will be added to input data from highest dimension, for example:
                1. if the `weights` only has 1 dimension, it will be added to the `E` dimension of input data.
                2. if the `weights` has 3 dimensions, it will be added to `E`, `B` and `C` dimensions.
                it's a typical practice to add weights for different classes:
                to ensemble 3 segmentation model outputs, every output has 4 channels(classes),
                so the input data shape can be: [3, B, 4, H, W, D].
                and add different `weights` for different classes, so the `weights` shape can be: [3, 1, 4].
                for example: `weights = [[[1, 2, 3, 4]], [[4, 3, 2, 1]], [[1, 1, 1, 1]]]`.

        """
        ensemble = MeanEnsemble(weights=weights)
        super().__init__(keys, ensemble, output_key)


class VoteEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.VoteEnsemble`.
    """

    def __init__(
        self, keys: KeysCollection, output_key: Optional[str] = None, num_classes: Optional[int] = None
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.
            num_classes: if the input is single channel data instead of One-Hot, we can't get class number
                from channel, need to explicitly specify the number of classes to vote.

        """
        ensemble = VoteEnsemble(num_classes=num_classes)
        super().__init__(keys, ensemble, output_key)


class Decollated(MapTransform):
    """
    Decollate a batch of data.

    Note that unlike most MapTransforms, this will decollate all data, so keys are not needed.

    Args:
        batch_size: if not supplied, we try to determine it based on array lengths. Will raise an error if
            it fails to determine it automatically.
    """

    def __init__(self, batch_size: Optional[int] = None) -> None:
        super().__init__(None)
        self.batch_size = batch_size

    def __call__(self, data: dict) -> List[dict]:
        return monai.data.decollate_batch(data, self.batch_size)


class ProbNMSd(MapTransform):
    """
    Performs probability based non-maximum suppression (NMS) on the probabilities map via
    iteratively selecting the coordinate with highest probability and then move it as well
    as its surrounding values. The remove range is determined by the parameter `box_size`.
    If multiple coordinates have the same highest probability, only one of them will be
    selected.

    Args:
        spatial_dims: number of spatial dimensions of the input probabilities map.
            Defaults to 2.
        sigma: the standard deviation for gaussian filter.
            It could be a single value, or `spatial_dims` number of values. Defaults to 0.0.
        prob_threshold: the probability threshold, the function will stop searching if
            the highest probability is no larger than the threshold. The value should be
            no less than 0.0. Defaults to 0.5.
        box_size: the box size (in pixel) to be removed around the the pixel with the maximum probability.
            It can be an integer that defines the size of a square or cube,
            or a list containing different values for each dimensions. Defaults to 48.

    Return:
        a list of selected lists, where inner lists contain probability and coordinates.
        For example, for 3D input, the inner lists are in the form of [probability, x, y, z].

    Raises:
        ValueError: When ``prob_threshold`` is less than 0.0.
        ValueError: When ``box_size`` is a list or tuple, and its length is not equal to `spatial_dims`.
        ValueError: When ``box_size`` has a less than 1 value.

    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 2,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.0,
        prob_threshold: float = 0.5,
        box_size: Union[int, Sequence[int]] = 48,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.prob_nms = ProbNMS(
            spatial_dims=spatial_dims,
            sigma=sigma,
            prob_threshold=prob_threshold,
            box_size=box_size,
        )

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.prob_nms(d[key])
        return d


class Invertd(MapTransform):
    """
    Utility transform to automatically invert the previous transforms based on transform information.
    It extracts the transform information applied on the data with `orig_keys`, then inverts the transforms
    on the data with `keys`. several `keys` can share one `orig_keys`.
    A typical usage is to invert the pre-transforms (appplied on input `image`) on the model `pred` data.

    """

    def __init__(
        self,
        keys: KeysCollection,
        transform: InvertibleTransform,
        loader: TorchDataLoader,
        orig_keys: Union[str, Sequence[str]],
        meta_key_postfix: str = "meta_dict",
        collate_fn: Optional[Callable] = no_collation,
        postfix: str = "inverted",
        nearest_interp: Union[bool, Sequence[bool]] = True,
        to_tensor: Union[bool, Sequence[bool]] = True,
        device: Union[Union[str, torch.device], Sequence[Union[str, torch.device]]] = "cpu",
        post_func: Union[Callable, Sequence[Callable]] = lambda x: x,
        num_workers: Optional[int] = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: the key of expected data in the dict, invert transforms on it.
                it also can be a list of keys, will invert transform for each of them, like: ["pred", "pred_class2"].
            transform: the previous callable transform that applied on input data.
            loader: data loader used to run transforms and generate the batch of data.
            orig_keys: the key of the original input data in the dict. will get the applied transform information
                for this input data, then invert them for the expected data with `keys`.
                It can also be a list of keys, each matches to the `keys` data.
            meta_key_postfix: use `{orig_key}_{postfix}` to to fetch the meta data from dict,
                default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle orig_key `image`,  read/write `affine` matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.
            collate_fn: how to collate data after inverse transformations.
                default won't do any collation, so the output will be a list of size batch size.
            postfix: will save the inverted result into dict with key `{key}_{postfix}`.
            nearest_interp: whether to use `nearest` interpolation mode when inverting the spatial transforms,
                default to `True`. If `False`, use the same interpolation mode as the original transform.
                it also can be a list of bool, each matches to the `keys` data.
            to_tensor: whether to convert the inverted data into PyTorch Tensor first, default to `True`.
                it also can be a list of bool, each matches to the `keys` data.
            device: if converted to Tensor, move the inverted results to target device before `post_func`,
                default to "cpu", it also can be a list of string or `torch.device`,
                each matches to the `keys` data.
            post_func: post processing for the inverted data, should be a callable function.
                it also can be a list of callable, each matches to the `keys` data.
            num_workers: number of workers when run data loader for inverse transforms,
                default to 0 as only run one iteration and multi-processing may be even slower.
                Set to `None`, to use the `num_workers` of the input transform data loader.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.transform = transform
        self.inverter = BatchInverseTransform(
            transform=transform,
            loader=loader,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        self.orig_keys = ensure_tuple_rep(orig_keys, len(self.keys))
        self.meta_key_postfix = meta_key_postfix
        self.postfix = postfix
        self.nearest_interp = ensure_tuple_rep(nearest_interp, len(self.keys))
        self.to_tensor = ensure_tuple_rep(to_tensor, len(self.keys))
        self.device = ensure_tuple_rep(device, len(self.keys))
        self.post_func = ensure_tuple_rep(post_func, len(self.keys))
        self._totensor = ToTensor()

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key, orig_key, nearest_interp, to_tensor, device, post_func in self.key_iterator(
            d, self.orig_keys, self.nearest_interp, self.to_tensor, self.device, self.post_func
        ):
            transform_key = orig_key + InverseKeys.KEY_SUFFIX
            if transform_key not in d:
                warnings.warn(f"transform info of `{orig_key}` is not available or no InvertibleTransform applied.")
                continue

            transform_info = d[transform_key]
            if nearest_interp:
                transform_info = convert_inverse_interp_mode(
                    trans_info=deepcopy(transform_info),
                    mode="nearest",
                    align_corners=None,
                )

            input = d[key]
            if isinstance(input, torch.Tensor):
                input = input.detach()
            # construct the input dict data for BatchInverseTransform
            input_dict = {
                orig_key: input,
                transform_key: transform_info,
            }
            meta_dict_key = f"{orig_key}_{self.meta_key_postfix}"
            if meta_dict_key in d:
                input_dict[meta_dict_key] = d[meta_dict_key]

            with allow_missing_keys_mode(self.transform):  # type: ignore
                inverted = self.inverter(input_dict)

            # save the inverted data
            inverted_key = f"{key}_{self.postfix}"
            d[inverted_key] = [
                post_func(self._totensor(i[orig_key]).to(device) if to_tensor else i[orig_key]) for i in inverted
            ]

            # save the inverted meta dict
            if meta_dict_key in d:
                d[f"{inverted_key}_{self.meta_key_postfix}"] = [i.get(meta_dict_key) for i in inverted]
        return d


ActivationsD = ActivationsDict = Activationsd
AsDiscreteD = AsDiscreteDict = AsDiscreted
KeepLargestConnectedComponentD = KeepLargestConnectedComponentDict = KeepLargestConnectedComponentd
LabelToContourD = LabelToContourDict = LabelToContourd
MeanEnsembleD = MeanEnsembleDict = MeanEnsembled
ProbNMSD = ProbNMSDict = ProbNMSd
VoteEnsembleD = VoteEnsembleDict = VoteEnsembled
DecollateD = DecollateDict = Decollated
InvertD = InvertDict = Invertd

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
"""
A collection of dictionary-based wrappers around the "vanilla" transforms for model output tensors
defined in :py:class:`monai.transforms.utility.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from monai import config
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor, PathLike
from monai.data.csv_saver import CSVSaver
from monai.data.meta_obj import get_meta_dict_name
from monai.data.meta_tensor import MetaTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.post.array import (
    Activations,
    AsDiscrete,
    DistanceTransformEDT,
    FillHoles,
    KeepLargestConnectedComponent,
    LabelFilter,
    LabelToContour,
    MeanEnsemble,
    ProbNMS,
    RemoveSmallObjects,
    SobelGradients,
    VoteEnsemble,
)
from monai.transforms.transform import MapTransform
from monai.transforms.utility.array import ToTensor
from monai.transforms.utils import allow_missing_keys_mode, convert_applied_interp_mode
from monai.utils import PostFix, convert_to_tensor, ensure_tuple, ensure_tuple_rep

__all__ = [
    "ActivationsD",
    "ActivationsDict",
    "Activationsd",
    "AsDiscreteD",
    "AsDiscreteDict",
    "AsDiscreted",
    "Ensembled",
    "EnsembleD",
    "EnsembleDict",
    "FillHolesD",
    "FillHolesDict",
    "FillHolesd",
    "InvertD",
    "InvertDict",
    "Invertd",
    "KeepLargestConnectedComponentD",
    "KeepLargestConnectedComponentDict",
    "KeepLargestConnectedComponentd",
    "RemoveSmallObjectsD",
    "RemoveSmallObjectsDict",
    "RemoveSmallObjectsd",
    "LabelFilterD",
    "LabelFilterDict",
    "LabelFilterd",
    "LabelToContourD",
    "LabelToContourDict",
    "LabelToContourd",
    "MeanEnsembleD",
    "MeanEnsembleDict",
    "MeanEnsembled",
    "ProbNMSD",
    "ProbNMSDict",
    "ProbNMSd",
    "SaveClassificationD",
    "SaveClassificationDict",
    "SaveClassificationd",
    "SobelGradientsD",
    "SobelGradientsDict",
    "SobelGradientsd",
    "VoteEnsembleD",
    "VoteEnsembleDict",
    "VoteEnsembled",
    "DistanceTransformEDTd",
    "DistanceTransformEDTD",
    "DistanceTransformEDTDict",
]

DEFAULT_POST_FIX = PostFix.meta()


class Activationsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddActivations`.
    Add activation layers to the input data specified by `keys`.
    """

    backend = Activations.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigmoid: Sequence[bool] | bool = False,
        softmax: Sequence[bool] | bool = False,
        other: Sequence[Callable] | Callable | None = None,
        allow_missing_keys: bool = False,
        **kwargs,
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
            kwargs: additional parameters to `torch.softmax` (used when ``softmax=True``).
                Defaults to ``dim=0``, unrecognized parameters will be ignored.

        """
        super().__init__(keys, allow_missing_keys)
        self.sigmoid = ensure_tuple_rep(sigmoid, len(self.keys))
        self.softmax = ensure_tuple_rep(softmax, len(self.keys))
        self.other = ensure_tuple_rep(other, len(self.keys))
        self.converter = Activations()
        self.converter.kwargs = kwargs

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, sigmoid, softmax, other in self.key_iterator(d, self.sigmoid, self.softmax, self.other):
            d[key] = self.converter(d[key], sigmoid, softmax, other)
        return d


class AsDiscreted(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsDiscrete`.
    """

    backend = AsDiscrete.backend

    def __init__(
        self,
        keys: KeysCollection,
        argmax: Sequence[bool] | bool = False,
        to_onehot: Sequence[int | None] | int | None = None,
        threshold: Sequence[float | None] | float | None = None,
        rounding: Sequence[str | None] | str | None = None,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            argmax: whether to execute argmax function on input data before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            threshold: if not None, threshold the float values to int number 0 or 1 with specified threshold value.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            rounding: if not None, round the data according to the specified option,
                available options: ["torchrounding"]. it also can be a sequence of str or None,
                each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.
            kwargs: additional parameters to ``AsDiscrete``.
                ``dim``, ``keepdim``, ``dtype`` are supported, unrecognized parameters will be ignored.
                These default to ``0``, ``True``, ``torch.float`` respectively.

        """
        super().__init__(keys, allow_missing_keys)
        self.argmax = ensure_tuple_rep(argmax, len(self.keys))
        self.to_onehot = []
        for flag in ensure_tuple_rep(to_onehot, len(self.keys)):
            if isinstance(flag, bool):
                raise ValueError("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
            self.to_onehot.append(flag)

        self.threshold = []
        for flag in ensure_tuple_rep(threshold, len(self.keys)):
            if isinstance(flag, bool):
                raise ValueError("`threshold_values=True/False` is deprecated, please use `threshold=value` instead.")
            self.threshold.append(flag)

        self.rounding = ensure_tuple_rep(rounding, len(self.keys))
        self.converter = AsDiscrete()
        self.converter.kwargs = kwargs

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, argmax, to_onehot, threshold, rounding in self.key_iterator(
            d, self.argmax, self.to_onehot, self.threshold, self.rounding
        ):
            d[key] = self.converter(d[key], argmax, to_onehot, threshold, rounding)
        return d


class KeepLargestConnectedComponentd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.KeepLargestConnectedComponent`.
    """

    backend = KeepLargestConnectedComponent.backend

    def __init__(
        self,
        keys: KeysCollection,
        applied_labels: Sequence[int] | int | None = None,
        is_onehot: bool | None = None,
        independent: bool = True,
        connectivity: int | None = None,
        num_components: int = 1,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            applied_labels: Labels for applying the connected component analysis on.
                If given, voxels whose value is in this list will be analyzed.
                If `None`, all non-zero values will be analyzed.
            is_onehot: if `True`, treat the input data as OneHot format data, otherwise, not OneHot format data.
                default to None, which treats multi-channel data as OneHot and single channel data as not OneHot.
            independent: whether to treat ``applied_labels`` as a union of foreground labels.
                If ``True``, the connected component analysis will be performed on each foreground label independently
                and return the intersection of the largest components.
                If ``False``, the analysis will be performed on the union of foreground labels.
                default is `True`.
            connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. If ``None``, a full
                connectivity of ``input.ndim`` is used. for more details:
                https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label.
            num_components: The number of largest components to preserve.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.converter = KeepLargestConnectedComponent(
            applied_labels=applied_labels,
            is_onehot=is_onehot,
            independent=independent,
            connectivity=connectivity,
            num_components=num_components,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class RemoveSmallObjectsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RemoveSmallObjectsd`.

    Args:
        min_size: objects smaller than this size (in number of voxels; or surface area/volume value
            in whatever units your image is if by_measure is True) are removed.
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
            Accepted values are ranging from  1 to input.ndim. If ``None``, a full
            connectivity of ``input.ndim`` is used. For more details refer to linked scikit-image
            documentation.
        independent_channels: Whether or not to consider channels as independent. If true, then
            conjoining islands from different labels will be removed if they are below the threshold.
            If false, the overall size islands made from all non-background voxels will be used.
        by_measure: Whether the specified min_size is in number of voxels. if this is True then min_size
            represents a surface area or volume value of whatever units your image is in (mm^3, cm^2, etc.)
            default is False. e.g. if min_size is 3, by_measure is True and the units of your data is mm,
            objects smaller than 3mm^3 are removed.
        pixdim: the pixdim of the input image. if a single number, this is used for all axes.
            If a sequence of numbers, the length of the sequence must be equal to the image dimensions.
    """

    backend = RemoveSmallObjects.backend

    def __init__(
        self,
        keys: KeysCollection,
        min_size: int = 64,
        connectivity: int = 1,
        independent_channels: bool = True,
        by_measure: bool = False,
        pixdim: Sequence[float] | float | np.ndarray | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = RemoveSmallObjects(min_size, connectivity, independent_channels, by_measure, pixdim)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class LabelFilterd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LabelFilter`.
    """

    backend = LabelFilter.backend

    def __init__(
        self, keys: KeysCollection, applied_labels: Sequence[int] | int, allow_missing_keys: bool = False
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            applied_labels: Label(s) to filter on.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.converter = LabelFilter(applied_labels)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class FillHolesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.FillHoles`.
    """

    backend = FillHoles.backend

    def __init__(
        self,
        keys: KeysCollection,
        applied_labels: Iterable[int] | int | None = None,
        connectivity: int | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Initialize the connectivity and limit the labels for which holes are filled.

        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            applied_labels (Optional[Union[Iterable[int], int]], optional): Labels for which to fill holes. Defaults to None,
                that is filling holes for all labels.
            connectivity (int, optional): Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. Defaults to a full
                connectivity of ``input.ndim``.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = FillHoles(applied_labels=applied_labels, connectivity=connectivity)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class LabelToContourd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LabelToContour`.
    """

    backend = LabelToContour.backend

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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class Ensembled(MapTransform):
    """
    Base class of dictionary-based ensemble transforms.

    """

    backend = list(set(VoteEnsemble.backend) & set(MeanEnsemble.backend))

    def __init__(
        self,
        keys: KeysCollection,
        ensemble: Callable[[Sequence[NdarrayOrTensor] | NdarrayOrTensor], NdarrayOrTensor],
        output_key: str | None = None,
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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        items: list[NdarrayOrTensor] | NdarrayOrTensor
        if len(self.keys) == 1 and self.keys[0] in d:
            items = d[self.keys[0]]
        else:
            items = [d[key] for key in self.key_iterator(d)]

        if len(items) > 0:
            d[self.output_key] = self.ensemble(items)

        return d


class MeanEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MeanEnsemble`.
    """

    backend = MeanEnsemble.backend

    def __init__(
        self,
        keys: KeysCollection,
        output_key: str | None = None,
        weights: Sequence[float] | NdarrayOrTensor | None = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be stack and execute ensemble.
                if only 1 key provided, suppose it's a PyTorch Tensor with data stacked on dimension `E`.
            output_key: the key to store ensemble result in the dictionary.
                if only 1 key provided in `keys`, `output_key` can be None and use `keys` as default.
            weights: can be a list or tuple of numbers for input data with shape: [E, C, H, W[, D]].
                or a Numpy ndarray or a PyTorch Tensor data.
                the `weights` will be added to input data from highest dimension, for example:
                1. if the `weights` only has 1 dimension, it will be added to the `E` dimension of input data.
                2. if the `weights` has 2 dimensions, it will be added to `E` and `C` dimensions.
                it's a typical practice to add weights for different classes:
                to ensemble 3 segmentation model outputs, every output has 4 channels(classes),
                so the input data shape can be: [3, 4, H, W, D].
                and add different `weights` for different classes, so the `weights` shape can be: [3, 4].
                for example: `weights = [[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]]`.

        """
        ensemble = MeanEnsemble(weights=weights)
        super().__init__(keys, ensemble, output_key)


class VoteEnsembled(Ensembled):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.VoteEnsemble`.
    """

    backend = VoteEnsemble.backend

    def __init__(self, keys: KeysCollection, output_key: str | None = None, num_classes: int | None = None) -> None:
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
        box_size: the box size (in pixel) to be removed around the pixel with the maximum probability.
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

    backend = ProbNMS.backend

    def __init__(
        self,
        keys: KeysCollection,
        spatial_dims: int = 2,
        sigma: Sequence[float] | float | Sequence[torch.Tensor] | torch.Tensor = 0.0,
        prob_threshold: float = 0.5,
        box_size: int | Sequence[int] = 48,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.prob_nms = ProbNMS(
            spatial_dims=spatial_dims, sigma=sigma, prob_threshold=prob_threshold, box_size=box_size
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.prob_nms(d[key])
        return d


class Invertd(MapTransform):
    """
    Utility transform to invert the previously applied transforms.

    Taking the ``transform`` previously applied on ``orig_keys``, this ``Invertd`` will apply the inverse of it
    to the data stored at ``keys``.

    ``Invertd``'s output will also include a copy of the metadata
    dictionary (originally from  ``orig_meta_keys`` or the metadata of ``orig_keys``),
    with the relevant fields inverted and stored at ``meta_keys``.

    A typical usage is to apply the inverse of the preprocessing (``transform=preprocessings``) on
    input ``orig_keys=image`` to the model predictions ``keys=pred``.

    A detailed usage example is available in the tutorial:
    https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/torch/unet_inference_dict.py

    Note:

        - The output of the inverted data and metadata will be stored at ``keys`` and ``meta_keys`` respectively.
        - To correctly invert the transforms, the information of the previously applied transforms should be
          available at ``{orig_keys}_transforms``, and the original metadata at ``orig_meta_keys``.
          (``meta_key_postfix`` is an optional string to conveniently construct "meta_keys" and/or "orig_meta_keys".)
          see also: :py:class:`monai.transforms.TraceableTransform`.
        - The transform will not change the content in ``orig_keys`` and ``orig_meta_key``.
          These keys are only used to represent the data status of ``key`` before inverting.

    """

    def __init__(
        self,
        keys: KeysCollection,
        transform: InvertibleTransform,
        orig_keys: KeysCollection | None = None,
        meta_keys: KeysCollection | None = None,
        orig_meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        nearest_interp: bool | Sequence[bool] = True,
        to_tensor: bool | Sequence[bool] = True,
        device: str | torch.device | Sequence[str | torch.device] | None = None,
        post_func: Callable | Sequence[Callable] | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: the key of expected data in the dict, the inverse of ``transforms`` will be applied on it in-place.
                It also can be a list of keys, will apply the inverse transform respectively.
            transform: the transform applied to ``orig_key``, its inverse will be applied on ``key``.
            orig_keys: the key of the original input data in the dict. These keys default to `self.keys` if not set.
                the transform trace information of ``transforms`` should be stored at ``{orig_keys}_transforms``.
                It can also be a list of keys, each matches the ``keys``.
            meta_keys: The key to output the inverted metadata dictionary.
                The metadata is a dictionary optionally containing: filename, original_shape.
                It can be a sequence of strings, maps to ``keys``.
                If None, will try to create a metadata dict with the default key: `{key}_{meta_key_postfix}`.
            orig_meta_keys: the key of the metadata of original input data.
                The metadata is a dictionary optionally containing: filename, original_shape.
                It can be a sequence of strings, maps to the `keys`.
                If None, will try to create a metadata dict with the default key: `{orig_key}_{meta_key_postfix}`.
                This metadata dict will also be included in the inverted dict, stored in `meta_keys`.
            meta_key_postfix: if `orig_meta_keys` is None, use `{orig_key}_{meta_key_postfix}` to fetch the
                metadata from dict, if `meta_keys` is None, use `{key}_{meta_key_postfix}`. Default: ``"meta_dict"``.
            nearest_interp: whether to use `nearest` interpolation mode when inverting the spatial transforms,
                default to `True`. If `False`, use the same interpolation mode as the original transform.
                It also can be a list of bool, each matches to the `keys` data.
            to_tensor: whether to convert the inverted data into PyTorch Tensor first, default to `True`.
                It also can be a list of bool, each matches to the `keys` data.
            device: if converted to Tensor, move the inverted results to target device before `post_func`,
                default to None, it also can be a list of string or `torch.device`, each matches to the `keys` data.
            post_func: post processing for the inverted data, should be a callable function.
                It also can be a list of callable, each matches to the `keys` data.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        if not isinstance(transform, InvertibleTransform):
            raise ValueError("transform is not invertible, can't invert transform for the data.")
        self.transform = transform
        self.orig_keys = ensure_tuple_rep(orig_keys, len(self.keys)) if orig_keys is not None else self.keys
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.orig_meta_keys = ensure_tuple_rep(orig_meta_keys, len(self.keys))
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.nearest_interp = ensure_tuple_rep(nearest_interp, len(self.keys))
        self.to_tensor = ensure_tuple_rep(to_tensor, len(self.keys))
        self.device = ensure_tuple_rep(device, len(self.keys))
        self.post_func = ensure_tuple_rep(post_func, len(self.keys))
        self._totensor = ToTensor()

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        d = dict(data)
        for (
            key,
            orig_key,
            meta_key,
            orig_meta_key,
            meta_key_postfix,
            nearest_interp,
            to_tensor,
            device,
            post_func,
        ) in self.key_iterator(
            d,
            self.orig_keys,
            self.meta_keys,
            self.orig_meta_keys,
            self.meta_key_postfix,
            self.nearest_interp,
            self.to_tensor,
            self.device,
            self.post_func,
        ):
            if isinstance(d[key], MetaTensor):
                if orig_key not in d:
                    warnings.warn(f"transform info of `{orig_key}` is not available in MetaTensor {key}.")
                    continue
            else:
                transform_key = InvertibleTransform.trace_key(orig_key)
                if transform_key not in d:
                    warnings.warn(f"transform info of `{orig_key}` is not available or no InvertibleTransform applied.")
                    continue

            orig_meta_key = orig_meta_key or f"{orig_key}_{meta_key_postfix}"
            if orig_key in d and isinstance(d[orig_key], MetaTensor):
                transform_info = d[orig_key].applied_operations
                meta_info = d[orig_key].meta
            else:
                transform_info = d[InvertibleTransform.trace_key(orig_key)]
                meta_info = d.get(orig_meta_key, {})
            if nearest_interp:
                transform_info = convert_applied_interp_mode(
                    trans_info=transform_info, mode="nearest", align_corners=None
                )

            inputs = d[key]
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.detach()

            if not isinstance(inputs, MetaTensor):
                inputs = convert_to_tensor(inputs, track_meta=True)
            inputs.applied_operations = deepcopy(transform_info)
            inputs.meta = deepcopy(meta_info)

            # construct the input dict data
            input_dict = {orig_key: inputs}
            if config.USE_META_DICT:
                input_dict[InvertibleTransform.trace_key(orig_key)] = transform_info
                input_dict[PostFix.meta(orig_key)] = meta_info
            with allow_missing_keys_mode(self.transform):  # type: ignore
                inverted = self.transform.inverse(input_dict)

            # save the inverted data
            inverted_data = inverted[orig_key]
            if to_tensor and not isinstance(inverted_data, MetaTensor):
                inverted_data = self._totensor(inverted_data)
            if isinstance(inverted_data, np.ndarray) and device is not None and torch.device(device).type != "cpu":
                raise ValueError(f"Inverted data with type of 'numpy.ndarray' support device='cpu', got {device}.")
            if isinstance(inverted_data, torch.Tensor):
                inverted_data = inverted_data.to(device=device)
            d[key] = post_func(inverted_data) if callable(post_func) else inverted_data
            # save the invertd applied_operations if it's in the source dict
            if InvertibleTransform.trace_key(orig_key) in d:
                d[InvertibleTransform.trace_key(orig_key)] = inverted_data.applied_operations
            # save the inverted meta dict if it's in the source dict
            if orig_meta_key in d:
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                d[meta_key] = inverted.get(orig_meta_key)
        return d


class SaveClassificationd(MapTransform):
    """
    Save the classification results and metadata into CSV file or other storage.

    """

    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        saver: CSVSaver | None = None,
        output_dir: PathLike = "./",
        filename: str = "predictions.csv",
        delimiter: str = ",",
        overwrite: bool = True,
        flush: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output, this transform only supports 1 key.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the metadata is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
                will extract the filename of input image to save classification results.
            meta_key_postfix: `key_{postfix}` was used to store the metadata in `LoadImaged`.
                so need the key to extract the metadata of input image, like filename, etc. default is `meta_dict`.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the metadata is a dictionary object which contains: filename, original_shape, etc.
                this arg only works when `meta_keys=None`. if no corresponding metadata, set to `None`.
            saver: the saver instance to save classification results, if None, create a CSVSaver internally.
                the saver must provide `save(data, meta_data)` and `finalize()` APIs.
            output_dir: if `saver=None`, specify the directory to save the CSV file.
            filename: if `saver=None`, specify the name of the saved CSV file.
            delimiter: the delimiter character in the saved file, default to "," as the default output type is `csv`.
                to be consistent with: https://docs.python.org/3/library/csv.html#csv.Dialect.delimiter.
            overwrite: if `saver=None`, indicate whether to overwriting existing CSV file content, if True,
                will clear the file before saving. otherwise, will append new content to the CSV file.
            flush: if `saver=None`, indicate whether to write the cache data to CSV file immediately
                in this transform and clear the cache. default to True.
                If False, may need user to call `saver.finalize()` manually or use `ClassificationSaver` handler.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        if len(self.keys) != 1:
            raise ValueError("only 1 key is allowed when saving the classification result.")
        self.saver = saver or CSVSaver(
            output_dir=output_dir, filename=filename, overwrite=overwrite, flush=flush, delimiter=delimiter
        )
        self.flush = flush
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(self, data):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            if meta_key is None and meta_key_postfix is not None:
                meta_key = f"{key}_{meta_key_postfix}"
            meta_data = d[meta_key] if meta_key is not None else None
            if meta_data is None:
                meta_data = d.get(get_meta_dict_name(key, d), None)
            self.saver.save(data=d[key], meta_data=meta_data)
            if self.flush:
                self.saver.finalize()

        return d

    def get_saver(self):
        """
        If want to write content into file, may need to call `finalize` of saver when epoch completed.
        Or users can also get the cache content from `saver` instead of writing into file.

        """
        return self.saver


class SobelGradientsd(MapTransform):
    """Calculate Sobel horizontal and vertical gradients of a grayscale image.

    Args:
        keys: keys of the corresponding items to model output.
        kernel_size: the size of the Sobel kernel. Defaults to 3.
        spatial_axes: the axes that define the direction of the gradient to be calculated. It calculate the gradient
            along each of the provide axis. By default it calculate the gradient for all spatial axes.
        normalize_kernels: if normalize the Sobel kernel to provide proper gradients. Defaults to True.
        normalize_gradients: if normalize the output gradient to 0 and 1. Defaults to False.
        padding_mode: the padding mode of the image when convolving with Sobel kernels. Defaults to `"reflect"`.
            Acceptable values are ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            See ``torch.nn.Conv1d()`` for more information.
        dtype: kernel data type (torch.dtype). Defaults to `torch.float32`.
        new_key_prefix: this prefix be prepended to the key to create a new key for the output and keep the value of
            key intact. By default not prefix is set and the corresponding array to the key will be replaced.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = SobelGradients.backend

    def __init__(
        self,
        keys: KeysCollection,
        kernel_size: int = 3,
        spatial_axes: Sequence[int] | int | None = None,
        normalize_kernels: bool = True,
        normalize_gradients: bool = False,
        padding_mode: str = "reflect",
        dtype: torch.dtype = torch.float32,
        new_key_prefix: str | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = SobelGradients(
            kernel_size=kernel_size,
            spatial_axes=spatial_axes,
            normalize_kernels=normalize_kernels,
            normalize_gradients=normalize_gradients,
            padding_mode=padding_mode,
            dtype=dtype,
        )
        self.new_key_prefix = new_key_prefix
        self.kernel_diff = self.transform.kernel_diff
        self.kernel_smooth = self.transform.kernel_smooth

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            new_key = key if self.new_key_prefix is None else self.new_key_prefix + key
            d[new_key] = self.transform(d[key])

        return d


class DistanceTransformEDTd(MapTransform):
    """
    Applies the Euclidean distance transform on the input.
    Either GPU based with CuPy / cuCIM or CPU based with scipy.
    To use the GPU implementation, make sure cuCIM is available and that the data is a `torch.tensor` on a GPU device.

    Note that the results of the libraries can differ, so stick to one if possible.
    For details, check out the `SciPy`_ and `cuCIM`_ documentation and / or :func:`monai.transforms.utils.distance_transform_edt`.


    Note on the input shape:
        Has to be a channel first array, must have shape: (num_channels, H, W [,D]).
        Can be of any type but will be converted into binary: 1 wherever image equates to True, 0 elsewhere.
        Input gets passed channel-wise to the distance-transform, thus results from this function will differ
        from directly calling ``distance_transform_edt()`` in CuPy or SciPy.

    Args:
        keys: keys of the corresponding items to be transformed.
        allow_missing_keys: don't raise exception if key is missing.
        sampling: Spacing of elements along each dimension. If a sequence, must be of length equal to the input rank -1;
            if a single number, this is used for all axes. If not specified, a grid spacing of unity is implied.

    .. _SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
    .. _cuCIM: https://docs.rapids.ai/api/cucim/nightly/api/#cucim.core.operations.morphology.distance_transform_edt


    """

    backend = DistanceTransformEDT.backend

    def __init__(
        self, keys: KeysCollection, allow_missing_keys: bool = False, sampling: None | float | list[float] = None
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.sampling = sampling
        self.distance_transform = DistanceTransformEDT(sampling=self.sampling)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.distance_transform(img=d[key])

        return d


ActivationsD = ActivationsDict = Activationsd
AsDiscreteD = AsDiscreteDict = AsDiscreted
FillHolesD = FillHolesDict = FillHolesd
InvertD = InvertDict = Invertd
KeepLargestConnectedComponentD = KeepLargestConnectedComponentDict = KeepLargestConnectedComponentd
RemoveSmallObjectsD = RemoveSmallObjectsDict = RemoveSmallObjectsd
LabelFilterD = LabelFilterDict = LabelFilterd
LabelToContourD = LabelToContourDict = LabelToContourd
MeanEnsembleD = MeanEnsembleDict = MeanEnsembled
ProbNMSD = ProbNMSDict = ProbNMSd
SaveClassificationD = SaveClassificationDict = SaveClassificationd
VoteEnsembleD = VoteEnsembleDict = VoteEnsembled
EnsembleD = EnsembleDict = Ensembled
SobelGradientsD = SobelGradientsDict = SobelGradientsd
DistanceTransformEDTD = DistanceTransformEDTDict = DistanceTransformEDTd

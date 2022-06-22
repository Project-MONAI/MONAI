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

import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Union

import torch

from monai.config.type_definitions import KeysCollection, NdarrayOrTensor, PathLike
from monai.data.csv_saver import CSVSaver
from monai.data.meta_tensor import MetaTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.post.array import (
    Activations,
    AsDiscrete,
    FillHoles,
    KeepLargestConnectedComponent,
    LabelFilter,
    LabelToContour,
    MeanEnsemble,
    ProbNMS,
    VoteEnsemble,
)
from monai.transforms.transform import MapTransform
from monai.transforms.utility.array import ToTensor
from monai.transforms.utils import allow_missing_keys_mode, convert_inverse_interp_mode
from monai.utils import convert_to_tensor, deprecated_arg, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix

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
    "VoteEnsembleD",
    "VoteEnsembleDict",
    "VoteEnsembled",
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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, sigmoid, softmax, other in self.key_iterator(d, self.sigmoid, self.softmax, self.other):
            d[key] = self.converter(d[key], sigmoid, softmax, other)
        return d


class AsDiscreted(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsDiscrete`.
    """

    backend = AsDiscrete.backend

    @deprecated_arg(name="n_classes", new_name="num_classes", since="0.6", msg_suffix="please use `to_onehot` instead.")
    @deprecated_arg("num_classes", since="0.7", msg_suffix="please use `to_onehot` instead.")
    @deprecated_arg("logit_thresh", since="0.7", msg_suffix="please use `threshold` instead.")
    @deprecated_arg(
        name="threshold_values", new_name="threshold", since="0.7", msg_suffix="please use `threshold` instead."
    )
    def __init__(
        self,
        keys: KeysCollection,
        argmax: Union[Sequence[bool], bool] = False,
        to_onehot: Union[Sequence[Optional[int]], Optional[int]] = None,
        threshold: Union[Sequence[Optional[float]], Optional[float]] = None,
        rounding: Union[Sequence[Optional[str]], Optional[str]] = None,
        allow_missing_keys: bool = False,
        n_classes: Optional[Union[Sequence[int], int]] = None,  # deprecated
        num_classes: Optional[Union[Sequence[int], int]] = None,  # deprecated
        logit_thresh: Union[Sequence[float], float] = 0.5,  # deprecated
        threshold_values: Union[Sequence[bool], bool] = False,  # deprecated
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            argmax: whether to execute argmax function on input data before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold value.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            rounding: if not None, round the data according to the specified option,
                available options: ["torchrounding"]. it also can be a sequence of str or None,
                each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.

        .. deprecated:: 0.6.0
            ``n_classes`` is deprecated, use ``to_onehot`` instead.

        .. deprecated:: 0.7.0
            ``num_classes`` is deprecated, use ``to_onehot`` instead.
            ``logit_thresh`` is deprecated, use ``threshold`` instead.
            ``threshold_values`` is deprecated, use ``threshold`` instead.

        """
        super().__init__(keys, allow_missing_keys)
        self.argmax = ensure_tuple_rep(argmax, len(self.keys))
        to_onehot_ = ensure_tuple_rep(to_onehot, len(self.keys))
        num_classes = ensure_tuple_rep(num_classes, len(self.keys))
        self.to_onehot = []
        for flag, val in zip(to_onehot_, num_classes):
            if isinstance(flag, bool):
                warnings.warn("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
                self.to_onehot.append(val if flag else None)
            else:
                self.to_onehot.append(flag)

        threshold_ = ensure_tuple_rep(threshold, len(self.keys))
        logit_thresh = ensure_tuple_rep(logit_thresh, len(self.keys))
        self.threshold = []
        for flag, val in zip(threshold_, logit_thresh):
            if isinstance(flag, bool):
                warnings.warn("`threshold_values=True/False` is deprecated, please use `threshold=value` instead.")
                self.threshold.append(val if flag else None)
            else:
                self.threshold.append(flag)

        self.rounding = ensure_tuple_rep(rounding, len(self.keys))
        self.converter = AsDiscrete()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
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
        applied_labels: Optional[Union[Sequence[int], int]] = None,
        is_onehot: Optional[bool] = None,
        independent: bool = True,
        connectivity: Optional[int] = None,
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
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.converter = KeepLargestConnectedComponent(
            applied_labels=applied_labels, is_onehot=is_onehot, independent=independent, connectivity=connectivity
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
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
        self, keys: KeysCollection, applied_labels: Union[Sequence[int], int], allow_missing_keys: bool = False
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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
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
        applied_labels: Optional[Union[Iterable[int], int]] = None,
        connectivity: Optional[int] = None,
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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
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
        ensemble: Callable[[Union[Sequence[NdarrayOrTensor], NdarrayOrTensor]], NdarrayOrTensor],
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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        items: Union[List[NdarrayOrTensor], NdarrayOrTensor]
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
        output_key: Optional[str] = None,
        weights: Optional[Union[Sequence[float], NdarrayOrTensor]] = None,
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

    backend = ProbNMS.backend

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
            spatial_dims=spatial_dims, sigma=sigma, prob_threshold=prob_threshold, box_size=box_size
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.prob_nms(d[key])
        return d


class Invertd(MapTransform):
    """
    Utility transform to automatically invert the previously applied transforms.

    Taking the ``transform`` previously applied on ``orig_keys``, this ``Invertd`` will apply the inverse of it
    to the data stored at ``keys``. ``Invertd``'s output will also include a copy of the metadata
    dictionary (originally from  ``orig_meta_keys``), with the relevant fields inverted and stored at ``meta_keys``.

    A typical usage is to apply the inverse of the preprocessing on input ``image`` to the model ``pred``.

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
        orig_keys: KeysCollection,
        meta_keys: Optional[KeysCollection] = None,
        orig_meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        nearest_interp: Union[bool, Sequence[bool]] = True,
        to_tensor: Union[bool, Sequence[bool]] = True,
        device: Union[Union[str, torch.device], Sequence[Union[str, torch.device]]] = "cpu",
        post_func: Union[Callable, Sequence[Callable]] = lambda x: x,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: the key of expected data in the dict, the inverse of ``transforms`` will be applied on it in-place.
                It also can be a list of keys, will apply the inverse transform respectively.
            transform: the transform applied to ``orig_key``, its inverse will be applied on ``key``.
            orig_keys: the key of the original input data in the dict.
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
                default to "cpu", it also can be a list of string or `torch.device`, each matches to the `keys` data.
            post_func: post processing for the inverted data, should be a callable function.
                It also can be a list of callable, each matches to the `keys` data.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        if not isinstance(transform, InvertibleTransform):
            raise ValueError("transform is not invertible, can't invert transform for the data.")
        self.transform = transform
        self.orig_keys = ensure_tuple_rep(orig_keys, len(self.keys))
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

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
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

            if orig_key in d and isinstance(d[orig_key], MetaTensor):
                transform_info = d[orig_key].applied_operations
                meta_info = d[orig_key].meta
            else:
                transform_info = d[InvertibleTransform.trace_key(orig_key)]
                meta_info = d.get(orig_meta_key or f"{orig_key}_{meta_key_postfix}", {})
            if nearest_interp:
                transform_info = convert_inverse_interp_mode(
                    trans_info=deepcopy(transform_info), mode="nearest", align_corners=None
                )

            inputs = d[key]
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.detach()

            if not isinstance(inputs, MetaTensor):
                inputs = convert_to_tensor(inputs, track_meta=True)
            inputs.applied_operations = transform_info
            inputs.meta = meta_info

            # construct the input dict data
            input_dict = {orig_key: inputs}

            with allow_missing_keys_mode(self.transform):  # type: ignore
                inverted = self.transform.inverse(input_dict)

            # save the inverted data
            if to_tensor and not isinstance(inverted[orig_key], MetaTensor):
                inverted_data = self._totensor(inverted[orig_key])
            else:
                inverted_data = inverted[orig_key]
            d[key] = post_func(inverted_data.to(device))

            # save the inverted meta dict
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
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        saver: Optional[CSVSaver] = None,
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


ActivationsD = ActivationsDict = Activationsd
AsDiscreteD = AsDiscreteDict = AsDiscreted
FillHolesD = FillHolesDict = FillHolesd
InvertD = InvertDict = Invertd
KeepLargestConnectedComponentD = KeepLargestConnectedComponentDict = KeepLargestConnectedComponentd
LabelFilterD = LabelFilterDict = LabelFilterd
LabelToContourD = LabelToContourDict = LabelToContourd
MeanEnsembleD = MeanEnsembleDict = MeanEnsembled
ProbNMSD = ProbNMSDict = ProbNMSd
SaveClassificationD = SaveClassificationDict = SaveClassificationd
VoteEnsembleD = VoteEnsembleDict = VoteEnsembled
EnsembleD = EnsembleDict = Ensembled

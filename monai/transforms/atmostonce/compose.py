import warnings
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from monai.transforms.atmostonce.apply import Apply
from monai.transforms.atmostonce.lazy_transform import LazyTransform, compile_lazy_transforms, flatten_sequences
from monai.transforms.atmostonce.utility import CachedTransformCompose, MultiSampleTransformCompose, \
    IMultiSampleTransform, IRandomizableTransform, ILazyTransform
from monai.utils import GridSampleMode, GridSamplePadMode, ensure_tuple, get_seed, MAX_SEED

from monai.transforms import Randomizable, InvertibleTransform, OneOf, apply_transform


# TODO: this is intended to replace Compose once development is done

# class ComposeCompiler:
#     """
#     Args:
#         transforms: A sequence of callable transforms
#         lazy_resampling: Whether to resample the data after each transform or accumulate
#             changes and then resample according to the accumulated changes as few times as
#             possible. Defaults to True as this nearly always improves speed and quality
#         caching_policy: Whether to cache deterministic transforms before the first
#             randomised transforms. This can be one of "off", "drive", "memory"
#         caching_favor: Whether to cache primarily for "speed" or for "quality". "speed" will
#             favor doing more work before caching, whereas "quality" will favour delaying
#             resampling until after caching
#     """
#     def __init__(
#             self,
#             transforms: Union[Sequence[Callable], Callable],
#             lazy_resampling: Optional[bool] = True,
#             caching_policy: Optional[str] = "off",
#             caching_favor: Optional[str] = "quality"
#     ):
#         valid_caching_policies = ("off", "drive", "memory")
#         if caching_policy not in valid_caching_policies:
#             raise ValueError("parameter 'caching_policy' must be one of "
#                              f"{valid_caching_policies} but is '{caching_policy}'")
#
#         dest_transforms = None
#
#         if caching_policy == "off":
#             if lazy_resampling is False:
#                 dest_transforms = [t for t in transforms]
#             else:
#                 dest_transforms = ComposeCompiler.lazy_no_cache()
#         else:
#             if caching_policy == "drive":
#                 raise NotImplementedError()
#             elif caching_policy == "memory":
#                 raise NotImplementedError()
#
#         self.src_transforms = [t for t in transforms]
#         self.dest_transforms = dest_transforms
#
#     def __getitem__(
#             self,
#             index
#     ):
#         return self.dest_transforms[index]
#
#     def __len__(self):
#         return len(self.dest_transforms)
#
#     @staticmethod
#     def lazy_no_cache(transforms):
#         dest_transforms = []
#         # TODO: replace with lazy transform
#         cur_lazy = []
#         for i_t in range(1, len(transforms)):
#             if isinstance(transforms[i_t], LazyTransform):
#                 # add this to the stack of transforms to be handled lazily
#                     cur_lazy.append(transforms[i_t])
#             else:
#                 if len(cur_lazy) > 0:
#                     dest_transforms.append(cur_lazy)
#                     # TODO: replace with lazy transform
#                     cur_lazy = []
#                 dest_transforms.append(transforms[i_t])
#         return dest_transforms


class ComposeCompiler:

    def compile(self, transforms, cache_mechanism):

        transforms_ = self.compile_caching(transforms, cache_mechanism)

        transforms__ = self.compile_multisampling(transforms_)

        transforms___ = self.compile_lazy_resampling(transforms__)

        return transforms___

    def compile_caching(self, transforms, cache_mechanism):
        # TODO: handle being passed a transform list with containers
        # given a list of transforms, determine where to add a cached transform object
        # and what transforms to put in it
        cacheable = list()
        for t in transforms:
            if self.transform_is_random(t) is False:
                cacheable.append(t)
            else:
                break

        if len(cacheable) == 0:
            return list(transforms)
        else:
            return [CachedTransformCompose(cacheable, cache_mechanism)] + transforms[len(cacheable):]

    def compile_multisampling(self, transforms):
        for i in reversed(range(len(transforms))):
            if self.transform_is_multisampling(transforms[i]) is True:
                transforms_ = transforms[:i] + [MultiSampleTransformCompose(transforms[i],
                                                                           transforms[i+1:])]
                return self.compile_multisampling(transforms_)

        return list(transforms)

    def compile_lazy_resampling(self, transforms):
        result = list()
        lazy = list()
        for i in range(len(transforms)):
            if self.transform_is_lazy(transforms[i]):
                lazy.append(transforms[i])
            else:
                if len(lazy) > 0:
                    result.extend(lazy)
                    result.append(Apply())
                    lazy = list()
                result.append(transforms[i])
        if len(lazy) > 0:
            result.extend(lazy)
            result.append(Apply())
        return result

    def transform_is_random(self, t):
        return isinstance(t, IRandomizableTransform)

    def transform_is_container(self, t):
        return isinstance(t, CachedTransformCompose, MultiSampleTransformCompose)

    def transform_is_multisampling(self, t):
        return isinstance(t, IMultiSampleTransform)

    def transform_is_lazy(self, t):
        return isinstance(t, ILazyTransform)


class Compose(Randomizable, InvertibleTransform):
    """
    ``Compose`` provides the ability to chain a series of callables together in
    a sequential manner. Each transform in the sequence must take a single
    argument and return a single value.

    ``Compose`` can be used in two ways:

    #. With a series of transforms that accept and return a single
       ndarray / tensor / tensor-like parameter.
    #. With a series of transforms that accept and return a dictionary that
       contains one or more parameters. Such transforms must have pass-through
       semantics that unused values in the dictionary must be copied to the return
       dictionary. It is required that the dictionary is copied between input
       and output of each transform.

    If some transform takes a data item dictionary as input, and returns a
    sequence of data items in the transform chain, all following transforms
    will be applied to each item of this list if `map_items` is `True` (the
    default).  If `map_items` is `False`, the returned sequence is passed whole
    to the next callable in the chain.

    For example:

    A `Compose([transformA, transformB, transformC],
    map_items=True)(data_dict)` could achieve the following patch-based
    transformation on the `data_dict` input:

    #. transformA normalizes the intensity of 'img' field in the `data_dict`.
    #. transformB crops out image patches from the 'img' and 'seg' of
       `data_dict`, and return a list of three patch samples::

        {'img': 3x100x100 data, 'seg': 1x100x100 data, 'shape': (100, 100)}
                             applying transformB
                                 ---------->
        [{'img': 3x20x20 data, 'seg': 1x20x20 data, 'shape': (20, 20)},
         {'img': 3x20x20 data, 'seg': 1x20x20 data, 'shape': (20, 20)},
         {'img': 3x20x20 data, 'seg': 1x20x20 data, 'shape': (20, 20)},]

    #. transformC then randomly rotates or flips 'img' and 'seg' of
       each dictionary item in the list returned by transformB.

    The composed transforms will be set the same global random seed if user called
    `set_determinism()`.

    When using the pass-through dictionary operation, you can make use of
    :class:`monai.transforms.adaptors.adaptor` to wrap transforms that don't conform
    to the requirements. This approach allows you to use transforms from
    otherwise incompatible libraries with minimal additional work.

    Note:

        In many cases, Compose is not the best way to create pre-processing
        pipelines. Pre-processing is often not a strictly sequential series of
        operations, and much of the complexity arises when a not-sequential
        set of functions must be called as if it were a sequence.

        Example: images and labels
        Images typically require some kind of normalization that labels do not.
        Both are then typically augmented through the use of random rotations,
        flips, and deformations.
        Compose can be used with a series of transforms that take a dictionary
        that contains 'image' and 'label' entries. This might require wrapping
        `torchvision` transforms before passing them to compose.
        Alternatively, one can create a class with a `__call__` function that
        calls your pre-processing functions taking into account that not all of
        them are called on the labels.

    Args:
        transforms: sequence of callables.
        map_items: whether to apply transform to each item in the input `data` if `data` is a list or tuple.
            defaults to `True`.
        unpack_items: whether to unpack input `data` with `*` as parameters for the callable function of transform.
            defaults to `False`.
        log_stats: whether to log the detailed information of data and applied transform when error happened,
            for NumPy array and PyTorch Tensor, log the data shape and value range,
            for other metadata, log the values directly. default to `False`.
        lazy_resample: whether to compute consecutive spatial transforms resampling lazily. Default to False.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode when ``lazy_resample=True``. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When `USE_COMPILED` is `True`, this argument uses
            ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
            See also: https://docs.monai.io/en/stable/networks.html#grid-pull
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values when ``lazy_resample=True``. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

    """

    def __init__(
        self,
        transforms: Optional[Union[Sequence[Callable], Callable]] = None,
        map_items: bool = True,
        unpack_items: bool = False,
        log_stats: bool = False,
        mode=GridSampleMode.BILINEAR,
        padding_mode=GridSamplePadMode.BORDER,
        lazy_evaluation: bool = False
    ) -> None:
        if transforms is None:
            transforms = []
        self.transforms = ensure_tuple(transforms)

        if lazy_evaluation is True:
            self.dst_transforms = compile_lazy_transforms(self.transforms)
        else:
            self.dst_transforms = flatten_sequences(self.transforms)

        self.map_items = map_items
        self.unpack_items = unpack_items
        self.log_stats = log_stats
        self.mode = mode
        self.padding_mode = padding_mode
        self.lazy_evaluation = lazy_evaluation
        self.set_random_state(seed=get_seed())

        if self.lazy_evaluation:
            for t in self.dst_transforms:
                if isinstance(t, LazyTransform):
                    t.lazy_evaluation = True

    def set_random_state(self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None) -> "Compose":
        super().set_random_state(seed=seed, state=state)
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            _transform.set_random_state(seed=self.R.randint(MAX_SEED, dtype="uint32"))
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        for _transform in self.transforms:
            if not isinstance(_transform, Randomizable):
                continue
            try:
                _transform.randomize(data)
            except TypeError as type_error:
                tfm_name: str = type(_transform).__name__
                warnings.warn(
                    f'Transform "{tfm_name}" in Compose not randomized\n{tfm_name}.{type_error}.', RuntimeWarning
                )

    # TODO: this is a more general function that could be implemented elsewhere
    def flatten(self):
        """Return a Composition with a simple list of transforms, as opposed to any nested Compositions.

        e.g., `t1 = Compose([x, x, x, x, Compose([Compose([x, x]), x, x])]).flatten()`
        will result in the equivalent of `t1 = Compose([x, x, x, x, x, x, x, x])`.

        """
        new_transforms = []
        for t in self.transforms:
            if isinstance(t, Compose) and not isinstance(t, OneOf):
                new_transforms += t.flatten().transforms
            else:
                new_transforms.append(t)

        return Compose(new_transforms)

    def __len__(self):
        """Return number of transformations."""
        return len(self.flatten().transforms)

    def __call__(self, input_):
        for _transform in self.dst_transforms:
            input_ = apply_transform(_transform, input_, self.map_items, self.unpack_items, self.log_stats)
        return input_

    def inverse(self, data):
        invertible_transforms = [t for t in self.flatten().transforms if isinstance(t, InvertibleTransform)]
        if not invertible_transforms:
            warnings.warn("inverse has been called but no invertible transforms have been supplied")

        # loop backwards over transforms
        for t in reversed(invertible_transforms):
            data = apply_transform(t.inverse, data, self.map_items, self.unpack_items, self.log_stats)
        return data

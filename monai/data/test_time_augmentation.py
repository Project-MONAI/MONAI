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

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.utils import decollate_batch, pad_list_data_collate
from monai.transforms.compose import Compose
from monai.transforms.croppad.batch import PadListDataCollate
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.post.dictionary import Invertd
from monai.transforms.transform import Randomizable
from monai.transforms.utils_pytorch_numpy_unification import mode, stack
from monai.utils import CommonKeys, PostFix, optional_import

if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

__all__ = ["TestTimeAugmentation"]

DEFAULT_POST_FIX = PostFix.meta()


def _identity(x):
    return x


class TestTimeAugmentation:
    """
    Class for performing test time augmentations. This will pass the same image through the network multiple times.

    The user passes transform(s) to be applied to each realisation, and provided that at least one of those transforms
    is random, the network's output will vary. Provided that inverse transformations exist for all supplied spatial
    transforms, the inverse can be applied to each realisation of the network's output. Once in the same spatial
    reference, the results can then be combined and metrics computed.

    Test time augmentations are a useful feature for computing network uncertainty, as well as observing the network's
    dependency on the applied random transforms.

    Reference:
        Wang et al.,
        Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional
        neural networks,
        https://doi.org/10.1016/j.neucom.2019.01.103

    Args:
        transform: transform (or composed) to be applied to each realisation. At least one transform must be of type
            `Randomizable`. All random transforms must be of type `InvertibleTransform`.
        batch_size: number of realisations to infer at once.
        num_workers: how many subprocesses to use for data.
        inferrer_fn: function to use to perform inference.
        device: device on which to perform inference.
        image_key: key used to extract image from input dictionary.
        orig_key: the key of the original input data in the dict. will get the applied transform information
            for this input data, then invert them for the expected data with `image_key`.
        orig_meta_keys: the key of the metadata of original input data, will get the `affine`, `data_shape`, etc.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            if None, will try to construct meta_keys by `{orig_key}_{meta_key_postfix}`.
        meta_key_postfix: use `key_{postfix}` to fetch the metadata according to the key data,
            default is `meta_dict`, the metadata is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
            this arg only works when `meta_keys=None`.
        to_tensor: whether to convert the inverted data into PyTorch Tensor first, default to `True`.
        output_device: if converted the inverted data to Tensor, move the inverted results to target device
            before `post_func`, default to "cpu".
        post_func: post processing for the inverted data, should be a callable function.
        return_full_data: normally, metrics are returned (mode, mean, std, vvc). Setting this flag to `True`
            will return the full data. Dimensions will be same size as when passing a single image through
            `inferrer_fn`, with a dimension appended equal in size to `num_examples` (N), i.e., `[N,C,H,W,[D]]`.
        progress: whether to display a progress bar.

    Example:
        .. code-block:: python

            model = UNet(...).to(device)
            transform = Compose([RandAffined(keys, ...), ...])
            transform.set_random_state(seed=123)  # ensure deterministic evaluation

            tt_aug = TestTimeAugmentation(
                transform, batch_size=5, num_workers=0, inferrer_fn=model, device=device
            )
            mode, mean, std, vvc = tt_aug(test_data)
    """

    def __init__(
        self,
        transform: InvertibleTransform,
        batch_size: int,
        num_workers: int = 0,
        inferrer_fn: Callable = _identity,
        device: Union[str, torch.device] = "cpu",
        image_key=CommonKeys.IMAGE,
        orig_key=CommonKeys.LABEL,
        nearest_interp: bool = True,
        orig_meta_keys: Optional[str] = None,
        meta_key_postfix=DEFAULT_POST_FIX,
        to_tensor: bool = True,
        output_device: Union[str, torch.device] = "cpu",
        post_func: Callable = _identity,
        return_full_data: bool = False,
        progress: bool = True,
    ) -> None:
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.inferrer_fn = inferrer_fn
        self.device = device
        self.image_key = image_key
        self.return_full_data = return_full_data
        self.progress = progress
        self._pred_key = CommonKeys.PRED
        self.inverter = Invertd(
            keys=self._pred_key,
            transform=transform,
            orig_keys=orig_key,
            orig_meta_keys=orig_meta_keys,
            meta_key_postfix=meta_key_postfix,
            nearest_interp=nearest_interp,
            to_tensor=to_tensor,
            device=output_device,
            post_func=post_func,
        )

        # check that the transform has at least one random component, and that all random transforms are invertible
        self._check_transforms()

    def _check_transforms(self):
        """Should be at least 1 random transform, and all random transforms should be invertible."""
        ts = [self.transform] if not isinstance(self.transform, Compose) else self.transform.transforms
        randoms = np.array([isinstance(t, Randomizable) for t in ts])
        invertibles = np.array([isinstance(t, InvertibleTransform) for t in ts])
        # check at least 1 random
        if sum(randoms) == 0:
            warnings.warn(
                "TTA usually has at least a `Randomizable` transform or `Compose` contains `Randomizable` transforms."
            )
        # check that whenever randoms is True, invertibles is also true
        for r, i in zip(randoms, invertibles):
            if r and not i:
                warnings.warn(
                    f"Not all applied random transform(s) are invertible. Problematic transform: {type(r).__name__}"
                )

    def __call__(
        self, data: Dict[str, Any], num_examples: int = 10
    ) -> Union[Tuple[NdarrayOrTensor, NdarrayOrTensor, NdarrayOrTensor, float], NdarrayOrTensor]:
        """
        Args:
            data: dictionary data to be processed.
            num_examples: number of realisations to be processed and results combined.

        Returns:
            - if `return_full_data==False`: mode, mean, std, vvc. The mode, mean and standard deviation are
                calculated across `num_examples` outputs at each voxel. The volume variation coefficient (VVC)
                is `std/mean` across the whole output, including `num_examples`. See original paper for clarification.
            - if `return_full_data==False`: data is returned as-is after applying the `inferrer_fn` and then
                concatenating across the first dimension containing `num_examples`. This allows the user to perform
                their own analysis if desired.
        """
        d = dict(data)

        # check num examples is multiple of batch size
        if num_examples % self.batch_size != 0:
            raise ValueError("num_examples should be multiple of batch size.")

        # generate batch of data of size == batch_size, dataset and dataloader
        data_in = [deepcopy(d) for _ in range(num_examples)]
        ds = Dataset(data_in, self.transform)
        dl = DataLoader(ds, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=pad_list_data_collate)

        outs: List = []

        for b in tqdm(dl) if has_tqdm and self.progress else dl:
            # do model forward pass
            b[self._pred_key] = self.inferrer_fn(b[self.image_key].to(self.device))
            outs.extend([self.inverter(PadListDataCollate.inverse(i))[self._pred_key] for i in decollate_batch(b)])

        output: NdarrayOrTensor = stack(outs, 0)

        if self.return_full_data:
            return output

        # calculate metrics
        _mode = mode(output, dim=0)
        mean = output.mean(0)
        std = output.std(0)
        vvc = (output.std() / output.mean()).item()

        return _mode, mean, std, vvc

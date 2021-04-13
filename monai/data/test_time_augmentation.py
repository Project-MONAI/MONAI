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

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.inverse_batch_transform import BatchInverseTransform
from monai.data.utils import list_data_collate, pad_list_data_collate
from monai.transforms.compose import Compose
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import Randomizable
from monai.transforms.utils import allow_missing_keys_mode
from monai.utils.enums import CommonKeys, InverseKeys
from monai.utils.module import optional_import

if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

__all__ = ["TestTimeAugmentation"]


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
        label_key: key used to extract label from input dictionary.
        meta_key_postfix: use `key_{postfix}` to to fetch the meta data according to the key data,
            default is `meta_dict`, the meta data is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        return_full_data: normally, metrics are returned (mode, mean, std, vvc). Setting this flag to `True` will return the
            full data. Dimensions will be same size as when passing a single image through `inferrer_fn`, with a dimension appended
            equal in size to `num_examples` (N), i.e., `[N,C,H,W,[D]]`.
        progress: whether to display a progress bar.

    Example:
        .. code-block:: python

            transform = RandAffined(keys, ...)
            post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

            tt_aug = TestTimeAugmentation(
                transform, batch_size=5, num_workers=0, inferrer_fn=lambda x: post_trans(model(x)), device=device
            )
            mode, mean, std, vvc = tt_aug(test_data)
    """

    def __init__(
        self,
        transform: InvertibleTransform,
        batch_size: int,
        num_workers: int,
        inferrer_fn: Callable,
        device: Optional[Union[str, torch.device]] = "cuda" if torch.cuda.is_available() else "cpu",
        image_key=CommonKeys.IMAGE,
        label_key=CommonKeys.LABEL,
        meta_key_postfix="meta_dict",
        return_full_data: bool = False,
        progress: bool = True,
    ) -> None:
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.inferrer_fn = inferrer_fn
        self.device = device
        self.image_key = image_key
        self.label_key = label_key
        self.meta_key_postfix = meta_key_postfix
        self.return_full_data = return_full_data
        self.progress = progress

        # check that the transform has at least one random component, and that all random transforms are invertible
        self._check_transforms()

    def _check_transforms(self):
        """Should be at least 1 random transform, and all random transforms should be invertible."""
        ts = [self.transform] if not isinstance(self.transform, Compose) else self.transform.transforms
        randoms = np.array([isinstance(t, Randomizable) for t in ts])
        invertibles = np.array([isinstance(t, InvertibleTransform) for t in ts])
        # check at least 1 random
        if sum(randoms) == 0:
            raise RuntimeError(
                "Requires a `Randomizable` transform or a `Compose` containing at least one `Randomizable` transform."
            )
        # check that whenever randoms is True, invertibles is also true
        for r, i in zip(randoms, invertibles):
            if r and not i:
                raise RuntimeError(
                    f"All applied random transform(s) must be invertible. Problematic transform: {type(r).__name__}"
                )

    def __call__(
        self, data: Dict[str, Any], num_examples: int = 10
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]:
        """
        Args:
            data: dictionary data to be processed.
            num_examples: number of realisations to be processed and results combined.

        Returns:
            - if `return_full_data==False`: mode, mean, std, vvc. The mode, mean and standard deviation are calculated across
                `num_examples` outputs at each voxel. The volume variation coefficient (VVC) is `std/mean` across the whole output,
                including `num_examples`. See original paper for clarification.
            - if `return_full_data==False`: data is returned as-is after applying the `inferrer_fn` and then concatenating across
                the first dimension containing `num_examples`. This allows the user to perform their own analysis if desired.
        """
        d = dict(data)

        # check num examples is multiple of batch size
        if num_examples % self.batch_size != 0:
            raise ValueError("num_examples should be multiple of batch size.")

        # generate batch of data of size == batch_size, dataset and dataloader
        data_in = [d] * num_examples
        ds = Dataset(data_in, self.transform)
        dl = DataLoader(ds, self.num_workers, batch_size=self.batch_size, collate_fn=pad_list_data_collate)

        label_transform_key = self.label_key + InverseKeys.KEY_SUFFIX

        # create inverter
        inverter = BatchInverseTransform(self.transform, dl, collate_fn=list_data_collate)

        outputs: List[np.ndarray] = []

        for batch_data in tqdm(dl) if has_tqdm and self.progress else dl:

            batch_images = batch_data[self.image_key].to(self.device)

            # do model forward pass
            batch_output = self.inferrer_fn(batch_images)
            if isinstance(batch_output, torch.Tensor):
                batch_output = batch_output.detach().cpu()
            if isinstance(batch_output, np.ndarray):
                batch_output = torch.Tensor(batch_output)

            # create a dictionary containing the inferred batch and their transforms
            inferred_dict = {self.label_key: batch_output, label_transform_key: batch_data[label_transform_key]}
            # if meta dict is present, add that too (required for some inverse transforms)
            label_meta_dict_key = f"{self.label_key}_{self.meta_key_postfix}"
            if label_meta_dict_key in batch_data:
                inferred_dict[label_meta_dict_key] = batch_data[label_meta_dict_key]

            # do inverse transformation (allow missing keys as only inverting label)
            with allow_missing_keys_mode(self.transform):  # type: ignore
                inv_batch = inverter(inferred_dict)

            # append
            outputs.append(inv_batch[self.label_key])

        # output
        output: np.ndarray = np.concatenate(outputs)

        if self.return_full_data:
            return output

        # calculate metrics
        mode = np.array(torch.mode(torch.Tensor(output.astype(np.int64)), dim=0).values)
        mean: np.ndarray = np.mean(output, axis=0)  # type: ignore
        std: np.ndarray = np.std(output, axis=0)  # type: ignore
        vvc: float = (np.std(output) / np.mean(output)).item()
        return mode, mean, std, vvc

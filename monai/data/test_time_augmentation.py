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

import numpy as np
from typing import Any, Dict
import torch

from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.inverse_batch_transform import BatchInverseTransform
from monai.data.utils import pad_list_data_collate
from monai.transforms.compose import Compose
from monai.transforms.inverse_transform import InvertibleTransform
from monai.transforms.transform import Randomizable


__all__ = ["TestTimeAugmentation"]

def is_transform_rand(transform):
    if not isinstance(transform, Compose):
        return isinstance(transform, Randomizable)
    # call recursively for each sub-transform
    return any(is_transform_rand(t) for t in transform.transforms)


class TestTimeAugmentation:
    def __init__(
        self,
        transform: InvertibleTransform,
        batch_size,
        num_workers,
        inferrer_fn,
        device,
    ) -> None:
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.inferrer_fn = inferrer_fn
        self.device = device

        # check that the transform has at least one random component
        if not is_transform_rand(self.transform):
            raise RuntimeError(type(self).__name__ + " requires a `Randomizable` transform or a"
                               + " `Compose` containing at least one `Randomizable` transform.")

    def __call__(self, data: Dict[str, Any], num_examples=10, image_key="image", label_key="label", return_full_data=False):
        d = dict(data)

        # check num examples is multiple of batch size
        if num_examples % self.batch_size != 0:
            raise ValueError("num_examples should be multiple of batch size.")

        # generate batch of data of size == batch_size, dataset and dataloader
        data_in = [d for _ in range(num_examples)]
        ds = Dataset(data_in, self.transform)
        dl = DataLoader(ds, self.num_workers, batch_size=self.batch_size, collate_fn=pad_list_data_collate)

        label_transform_key = label_key + "_transforms"

        # create inverter
        inverter = BatchInverseTransform(self.transform, dl)

        outputs = []

        for batch_data in dl:

            batch_images = batch_data[image_key].to(self.device)

            # do model forward pass
            batch_output = self.inferrer_fn(batch_images)
            if isinstance(batch_output, torch.Tensor):
                batch_output = batch_output.detach().cpu()
            if isinstance(batch_output, np.ndarray):
                batch_output = torch.Tensor(batch_output)

            # check binary labels are extracted
            if not all(torch.unique(batch_output.int()) == torch.Tensor([0, 1])):
                raise RuntimeError("Test-time augmentation requires binary channels. If this is "
                                   "not binary segmentation, then you should one-hot your output.")

            # create a dictionary containing the inferred batch and their transforms
            inferred_dict = {label_key: batch_output, label_transform_key: batch_data[label_transform_key]}

            # do inverse transformation (only for the label key)
            inv_batch = inverter(inferred_dict, label_key)

            # append
            outputs.append(inv_batch)

        # calculate mean and standard deviation
        output = np.concatenate(outputs)

        if return_full_data:
            return output

        mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=output.astype(np.int64))
        mean = np.mean(output, axis=0)
        std = np.std(output, axis=0)
        vvc = np.std(output) / np.mean(output)
        return mode, mean, std, vvc

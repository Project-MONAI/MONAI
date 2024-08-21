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

import copy
import random

import monai
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from collections.abc import Callable, Sequence

__all__ = ["generate_prompt_pairs"]

ENABLE_SPECIAL = True
SPECIAL_INDEX = (23, 24, 25, 26, 27, 57, 128)
MERGE_LIST = {
    1: [25, 26],  # hepatic tumor and vessel merge into liver
    4: [24],  # pancreatic tumor merge into pancreas
    132: [57],  # overlap with trachea merge into airway
}


def get_point_label(id):
    # [B, N]
    if id in SPECIAL_INDEX and ENABLE_SPECIAL:
        return 2, 3
    else:
        return 0, 1
    
def generate_prompt_pairs(
    labels: Tensor,
    label_set: Sequence[int] | None = None,
    max_prompt: int | None = None,
    max_foreprompt: int | None = None,
    max_backprompt: int = 1,
    max_point: int = 20,
    include_background: bool = False,
    drop_label_prob: float = 0.2,
    drop_point_prob: float = 0.2,
    point_sampler: Callable | None = None
):
    """ Sample training pairs for VISTA3D training. 
    Args:
        labels: [1, 1, H, W, D], ground truth labels.
        label_set: the label list for the specific dataset.
        max_prompt: int, max number of total prompt, including foreground and background.
        max_foreprompt: int, max number of prompt from foreground.
        max_backprompt: int, max number of prompt from background.
        max_point: maximum number of points for each object.
        include_background: if include label=0 into training prompt. May casue issue in partial label
            trainig.
        drop_label_prob: probablity to drop label prompt.
        drop_point_prob: probablity to drop point prompt.
        point_sampler: sampler to augment masks with supervoxel.
    Returns:
        label_prompt: [B, 1]. The classes used for training automatic segmentation
        point: [B, N, 3]. The corresponding points for each class. Note that background label prompt
            requires matching point as well ([0,0,0] is used).
        point_label: [B, N]. The corresponding point labels for each point (negative or positive). 
            -1 is used for padding the background label prompt and will be ignored. 
        prompt_class: [B, 1], exactly the same with label_prompt for label indexing for training loss. 
            label_prompt can be None, and prompt_class is used to identify point classess.
    """
    # class label number
    assert labels.shape[0] == 1, "only support batch size 1"
    labels = labels[0, 0]
    device = labels.device
    unique_labels = labels.unique().cpu().numpy().tolist()
    if include_background:
        unique_labels = list(set(unique_labels) - (set(unique_labels) - set(label_set)))
    else:
        unique_labels = list(
            set(unique_labels) - (set(unique_labels) - set(label_set)) - set([0])
        )
    background_labels = list(set(label_set) - set(unique_labels))
    # during training, balance background and foreground prompts
    if max_backprompt is not None:
        if len(background_labels) > max_backprompt:
            random.shuffle(background_labels)
            background_labels = background_labels[:max_backprompt]

    if max_foreprompt is not None:
        if len(unique_labels) > max_foreprompt:
            random.shuffle(unique_labels)
            unique_labels = unique_labels[:max_foreprompt]

    if max_prompt is not None:
        if len(unique_labels) + len(background_labels) > max_prompt:
            if len(unique_labels) > max_prompt:
                unique_labels = random.sample(unique_labels, max_prompt)
                background_labels = []
            else:
                background_labels = random.sample(
                    background_labels, max_prompt - len(unique_labels)
                )
    _point = []
    _point_label = []
    # if use regular sampling
    if point_sampler is None:
        num_p = min(max_point, int(np.abs(random.gauss(mu=0, sigma=max_point // 2))) + 1)
        num_n = min(max_point, int(np.abs(random.gauss(mu=0, sigma=max_point // 2))))
        for id in unique_labels:
            neg_id, pos_id = get_point_label(id)
            plabels = labels == int(id)
            nlabels = ~plabels
            plabelpoints = torch.nonzero(plabels)
            nlabelpoints = torch.nonzero(nlabels)
            # final sampled positive points
            num_pa = min(len(plabelpoints), num_p)
            # final sampled negative points
            num_na = min(len(nlabelpoints), num_n)
            _point.append(
                torch.stack(
                    random.choices(plabelpoints, k=num_pa) + random.choices(nlabelpoints, k=num_na)
                    + [torch.tensor([0, 0, 0], device=device)] * (num_p + num_n - num_pa - num_na)
                )
            )
            _point_label.append(
                torch.tensor([pos_id] * num_pa + [neg_id] * num_na + [-1] * (num_p + num_n - num_pa - num_na)
                ).to(device)
            )
        for id in background_labels:
            # pad the background labels
            _point.append(torch.zeros(num_p + num_n, 3).to(device))  # all 0
            _point_label.append(torch.zeros(num_p + num_n).to(device) - 1)  # -1 not a point
    else:
        _point, _point_label = point_sampler(unique_labels, Np=max_point, Nn=0)
        for id in background_labels:
            # pad the background labels
            _point.append(torch.zeros(len(_point_label[0]), 3).to(device))  # all 0
            _point_label.append(
                torch.zeros(len(_point_label[0])).to(device) - 1
            )  # -1 not a point
    if len(unique_labels) == 0 and len(background_labels) == 0:
        # if max_backprompt is 0 and len(unique_labels), there is no effective prompt and the iteration must
        # be skipped. Handle this in trainer. 
        label_prompt, point, point_label, prompt_class = None, None, None, None
    else:
        label_prompt = torch.tensor(unique_labels + background_labels).unsqueeze(-1).to(device).long()
        point = torch.stack(_point)
        point_label = torch.stack(_point_label)
        prompt_class = copy.deepcopy(label_prompt)
    if random.uniform(0, 1) < drop_label_prob and len(unique_labels) > 0:
        label_prompt = None
        # If label prompt is dropped, there is no need to pad with points with label -1.
        pad = len(background_labels)
        point = point[: len(point) - pad]
        point_label = point_label[: len(point_label) - pad]
        prompt_class = prompt_class[: len(prompt_class) - pad]
    else:
        if random.uniform(0, 1) < drop_point_prob:
            point = None
            point_label = None
    return label_prompt, point, point_label, prompt_class

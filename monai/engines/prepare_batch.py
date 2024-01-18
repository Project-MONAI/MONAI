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

from __future__ import annotations

from typing import Any, Mapping

import torch

from monai.engines import PrepareBatch, default_prepare_batch
from monai.networks.schedulers import Scheduler


class DiffusionPrepareBatch(PrepareBatch):
    """
    This class is used as a callable for the `prepare_batch` parameter of engine classes for diffusion training.

    Assuming a supervised training process, it will generate a noise field using `get_noise` for an input image, and
    return the image and noise field as the image/target pair plus the noise field the kwargs under the key "noise".
    This assumes the inferer being used in conjunction with this class expects a "noise" parameter to be provided.

    If the `condition_name` is provided, this must refer to a key in the input dictionary containing the condition
    field to be passed to the inferer. This will appear in the keyword arguments under the key "condition".

    """

    def __init__(self, num_train_timesteps: int, condition_name: str | None = None) -> None:
        self.condition_name = condition_name
        self.num_train_timesteps = num_train_timesteps

    def get_noise(self, images: torch.Tensor) -> torch.Tensor:
        """Returns the noise tensor for input tensor `images`, override this for different noise distributions."""
        return torch.randn_like(images)

    def get_timesteps(self, images: torch.Tensor) -> torch.Tensor:
        """Get a timestep, by default this is a random integer between 0 and `self.num_train_timesteps`."""
        return torch.randint(0, self.num_train_timesteps, (images.shape[0],), device=images.device).long()

    def get_target(self, images: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Return the target for the loss function, this is the `noise` value by default."""
        return noise

    def __call__(
        self,
        batchdata: dict[str, torch.Tensor],
        device: str | torch.device | None = None,
        non_blocking: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple, dict[str, torch.Tensor]]:
        images, _ = default_prepare_batch(batchdata, device, non_blocking, **kwargs)
        noise = self.get_noise(images).to(device, non_blocking=non_blocking, **kwargs)
        timesteps = self.get_timesteps(images).to(device, non_blocking=non_blocking, **kwargs)

        target = self.get_target(images, noise, timesteps).to(device, non_blocking=non_blocking, **kwargs)
        infer_kwargs = {"noise": noise, "timesteps": timesteps}

        if self.condition_name is not None and isinstance(batchdata, Mapping):
            infer_kwargs["conditioning"] = batchdata[self.condition_name].to(
                device, non_blocking=non_blocking, **kwargs
            )

        # return input, target, arguments, and keyword arguments where noise is the target and also a keyword value
        return images, target, (), infer_kwargs


class VPredictionPrepareBatch(DiffusionPrepareBatch):
    """
    This class is used as a callable for the `prepare_batch` parameter of engine classes for diffusion training.

    Assuming a supervised training process, it will generate a noise field using `get_noise` for an input image, and
    from this compute the velocity using the provided scheduler. This value is used as the target in place of the
    noise field itself although the noise is field is in the kwargs under the key "noise". This assumes the inferer
    being used in conjunction with this class expects a "noise" parameter to be provided.

    If the `condition_name` is provided, this must refer to a key in the input dictionary containing the condition
    field to be passed to the inferer. This will appear in the keyword arguments under the key "condition".

    """

    def __init__(self, scheduler: Scheduler, num_train_timesteps: int, condition_name: str | None = None) -> None:
        super().__init__(num_train_timesteps=num_train_timesteps, condition_name=condition_name)
        self.scheduler = scheduler

    def get_target(self, images, noise, timesteps):
        return self.scheduler.get_velocity(images, noise, timesteps)

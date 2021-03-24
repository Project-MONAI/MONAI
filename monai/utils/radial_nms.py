from typing import Union

import numpy as np
import torch

from monai.networks.layers import GaussianFilter


class RadialNMS:
    def __init__(self, spatial_dims: int, sigma: float = 0.0, prob_threshold: float = 0.5, radius: int = 24):
        self.sigma = sigma
        self.spatial_dims = spatial_dims
        if self.sigma > 0:
            self.filter = GaussianFilter(spatial_dims=spatial_dims, sigma=sigma)
        self.prob_threshold = prob_threshold
        self.radius = radius

    def __call__(self, probs_map: Union[np.ndarray, torch.Tensor], level: int):
        if self.sigma > 0:
            if isinstance(probs_map, np.ndarray):
                probs_map = torch.as_tensor(probs_map, dtype=torch.float)
            device = (
                torch.device("cuda")
                if (probs_map.device == "cuda" and torch.cuda.is_available())
                else torch.device("cpu:0")
            )
            self.filter.to(device)
            probs_map = self.filter(probs_map)
            probs_map = probs_map.detach().cpu().numpy()

        probs_map_shape = probs_map.shape
        resolution = pow(2, level)

        outputs = []
        while np.max(probs_map) > self.prob_threshold:
            max_idx = np.unravel_index(probs_map.argmax(), probs_map_shape)
            prob_max = probs_map[max_idx]
            max_idx_arr = np.asarray(max_idx)
            coord_wsi = ((max_idx_arr + 0.5) * resolution).astype(int)
            outputs.append([prob_max] + list(coord_wsi))

            # achieve min index for each dimension
            idx_min_range = max_idx_arr - self.radius
            idx_min_range[idx_min_range < 0] = 0

            # achieve max index for each dimension
            idx_upper_bound = np.asarray(probs_map_shape)

            idx_max_range = max_idx_arr + self.radius
            idx_max_range[idx_max_range > idx_upper_bound] = idx_upper_bound[idx_max_range > idx_upper_bound]
            # set values during index ranges for each dimension to 0
            slices = tuple([slice(idx_min_range[i], idx_max_range[i]) for i in range(self.spatial_dims)])
            probs_map[slices] = 0

        return outputs

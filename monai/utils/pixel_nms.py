from typing import Sequence, Union

import numpy as np
import torch

from monai.networks.layers import GaussianFilter


class PixelNMS:
    """
    Performs pixel based non-maximum suppression (NMS) on the probabilities map via
    iteratively selecting the coordinate with highest probability and then move it as well
    as its surrounding values. The remove range is determined by the parameter `pixel_dis`.
    If multiple coordinates have the same highest probability, only one of them will be
    selected.

    Args:
        spatial_dims: number of spatial dimensions of the input probabilities map.
        sigma: the standard deviation for gaussian filter.
            It could be a single value, or `spatial_dims` number of values. Defaults to 0.
        prob_threshold: the probability threshold, the PixelNMS will stop searching if
            the highest probability is no larger than the threshold. Defaults to 0.5.
        pixel_dis: the distance in pixels that determines the surrounding area of the selected
            coordinate that will be removed before the next iteration. The area is a square (cube)
            for 2D (3D) input, and the edge length is 2 * pixel_dis. Defaults to 24.

    Return:
        a list of selected lists, where inner lists contain probability and coordinates.
        For example, for 3D input, the inner lists are in the form of [probability, x, y, z].

    """

    def __init__(
        self,
        spatial_dims: int,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0,
        prob_threshold: float = 0.5,
        pixel_dis: int = 24,
    ) -> None:
        self.sigma = sigma
        self.spatial_dims = spatial_dims
        if self.sigma != 0:
            self.filter = GaussianFilter(spatial_dims=spatial_dims, sigma=sigma)
        self.prob_threshold = prob_threshold
        self.pixel_dis = pixel_dis

    def __call__(
        self,
        probs_map: Union[np.ndarray, torch.Tensor],
        resolution_level: int = 0,
    ):
        """
        probs_map: the input probabilities map, it must have shape (H[, W, ...]).
        resolution_level: the level at which the original input is made. The returned
            coordinates will be converted according to this value.
        """
        if self.sigma != 0:
            if not isinstance(probs_map, torch.Tensor):
                probs_map = torch.as_tensor(probs_map, dtype=torch.float)
            self.filter.to(probs_map)
            probs_map = self.filter(probs_map)

        if isinstance(probs_map, torch.Tensor):
            probs_map = probs_map.detach().cpu().numpy()

        probs_map_shape = probs_map.shape
        resolution = pow(2, resolution_level)

        outputs = []
        while np.max(probs_map) > self.prob_threshold:
            max_idx = np.unravel_index(probs_map.argmax(), probs_map_shape)
            prob_max = probs_map[max_idx]
            max_idx_arr = np.asarray(max_idx)
            coord_wsi = ((max_idx_arr + 0.5) * resolution).astype(int)
            outputs.append([prob_max] + list(coord_wsi))

            idx_min_range = (max_idx_arr - self.pixel_dis).clip(0, None)
            idx_max_range = (max_idx_arr + self.pixel_dis).clip(None, probs_map_shape)
            # for each dimension, set values during index ranges to 0
            slices = tuple([slice(idx_min_range[i], idx_max_range[i]) for i in range(self.spatial_dims)])
            probs_map[slices] = 0

        return outputs

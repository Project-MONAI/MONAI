import numpy as np
from skimage import filters


class RadialNMS:
    """
    Compute non-maximal suppression from a predicion probality map
    Args:
        sigma
        prob_threshold
        radius
    """

    def __init__(self, sigma: float = 0.0, prob_threshold: float = 0.5, radius: int = 24) -> None:
        self.sigma = sigma
        self.prob_threshold = prob_threshold
        self.radius = radius

    def __call__(self, probs_map: np.ndarray, level: int, output_filename: str) -> None:
        if self.sigma > 0:
            probs_map = filters.gaussian(probs_map, sigma=self.sigma)

        x_shape, y_shape = probs_map.shape
        resolution = pow(2, level)

        with open(output_filename + ".csv", "w") as outfile:
            while np.max(probs_map) > self.prob_threshold:
                prob_max = probs_map.max()
                max_idx = np.where(probs_map == prob_max)
                x_mask, y_mask = max_idx[0][0], max_idx[1][0]
                x_wsi = int((x_mask + 0.5) * resolution)
                y_wsi = int((y_mask + 0.5) * resolution)
                outfile.write("{:0.5f},{},{}".format(prob_max, x_wsi, y_wsi) + "\n")

                x_min = x_mask - self.radius if x_mask - self.radius > 0 else 0
                x_max = x_mask + self.radius if x_mask + self.radius <= x_shape else x_shape
                y_min = y_mask - self.radius if y_mask - self.radius > 0 else 0
                y_max = y_mask + self.radius if y_mask + self.radius <= y_shape else y_shape

                for x in range(x_min, x_max):
                    for y in range(y_min, y_max):
                        probs_map[x, y] = 0

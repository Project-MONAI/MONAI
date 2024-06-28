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

import numpy as np
from numpy.typing import NDArray

from monai.utils import min_version, optional_import

__all__ = ["UltrasoundConfidenceMap"]

cv2, _ = optional_import("cv2")
csc_matrix, _ = optional_import("scipy.sparse", "1.7.1", min_version, "csc_matrix")
spsolve, _ = optional_import("scipy.sparse.linalg", "1.7.1", min_version, "spsolve")
cg, _ = optional_import("scipy.sparse.linalg", "1.7.1", min_version, "cg")
hilbert, _ = optional_import("scipy.signal", "1.7.1", min_version, "hilbert")
ruge_stuben_solver, _ = optional_import("pyamg", "5.0.0", min_version, "ruge_stuben_solver")


class UltrasoundConfidenceMap:
    """Compute confidence map from an ultrasound image.
    This transform uses the method introduced by Karamalis et al. in https://doi.org/10.1016/j.media.2012.07.005.
    It generates a confidence map by setting source and sink points in the image and computing the probability
    for random walks to reach the source for each pixel.

    The official code is available at:
    https://campar.in.tum.de/Main/AthanasiosKaramalisCode

    Args:
        alpha (float, optional): Alpha parameter. Defaults to 2.0.
        beta (float, optional): Beta parameter. Defaults to 90.0.
        gamma (float, optional): Gamma parameter. Defaults to 0.05.
        mode (str, optional): 'RF' or 'B' mode data. Defaults to 'B'.
        sink_mode (str, optional): Sink mode. Defaults to 'all'. If 'mask' is selected, a mask must be when calling
            the transform. Can be 'all', 'mid', 'min', or 'mask'.
        use_cg (bool, optional): Use Conjugate Gradient method for solving the linear system. Defaults to False.
        cg_tol (float, optional): Tolerance for the Conjugate Gradient method. Defaults to 1e-6.
            Will be used only if `use_cg` is True.
        cg_maxiter (int, optional): Maximum number of iterations for the Conjugate Gradient method. Defaults to 200.
            Will be used only if `use_cg` is True.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 90.0,
        gamma: float = 0.05,
        mode="B",
        sink_mode="all",
        use_cg=False,
        cg_tol=1e-6,
        cg_maxiter=200,
    ):
        # The hyperparameters for confidence map estimation
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mode = mode
        self.sink_mode = sink_mode
        self.use_cg = use_cg
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter

        # The precision to use for all computations
        self.eps = np.finfo("float64").eps

        # Store sink indices for external use
        self._sink_indices = np.array([], dtype="float64")

    def sub2ind(self, size: tuple[int, ...], rows: NDArray, cols: NDArray) -> NDArray:
        """Converts row and column subscripts into linear indices,
        basically the copy of the MATLAB function of the same name.
        https://www.mathworks.com/help/matlab/ref/sub2ind.html

        This function is Pythonic so the indices start at 0.

        Args:
            size Tuple[int]: Size of the matrix
            rows (NDArray): Row indices
            cols (NDArray): Column indices

        Returns:
            indices (NDArray): 1-D array of linear indices
        """
        indices: NDArray = rows + cols * size[0]
        return indices

    def get_seed_and_labels(
        self, data: NDArray, sink_mode: str = "all", sink_mask: NDArray | None = None
    ) -> tuple[NDArray, NDArray]:
        """Get the seed and label arrays for the max-flow algorithm

        Args:
            data: Input array
            sink_mode (str, optional): Sink mode. Defaults to 'all'.
            sink_mask (NDArray, optional): Sink mask. Defaults to None.

        Returns:
            Tuple[NDArray, NDArray]: Seed and label arrays
        """

        # Seeds and labels (boundary conditions)
        seeds = np.array([], dtype="float64")
        labels = np.array([], dtype="float64")

        # Indices for all columns
        sc = np.arange(data.shape[1], dtype="float64")

        # SOURCE ELEMENTS - 1st matrix row
        # Indices for 1st row, it will be broadcasted with sc
        sr_up = np.array([0])
        seed = self.sub2ind(data.shape, sr_up, sc).astype("float64")
        seed = np.unique(seed)
        seeds = np.concatenate((seeds, seed))

        # Label 1
        label = np.ones_like(seed)
        labels = np.concatenate((labels, label))

        # Create seeds for sink elements

        if sink_mode == "all":
            # All elements in the last row
            sr_down = np.ones_like(sc) * (data.shape[0] - 1)
            self._sink_indices = np.array([sr_down, sc], dtype="int32")
            seed = self.sub2ind(data.shape, sr_down, sc).astype("float64")

        elif sink_mode == "mid":
            # Middle element in the last row
            sc_down = np.array([data.shape[1] // 2])
            sr_down = np.ones_like(sc_down) * (data.shape[0] - 1)
            self._sink_indices = np.array([sr_down, sc_down], dtype="int32")
            seed = self.sub2ind(data.shape, sr_down, sc_down).astype("float64")

        elif sink_mode == "min":
            # Minimum element in the last row (excluding 10% from the edges)
            ten_percent = int(data.shape[1] * 0.1)
            min_val = np.min(data[-1, ten_percent:-ten_percent])
            min_idxs = np.where(data[-1, ten_percent:-ten_percent] == min_val)[0] + ten_percent
            sc_down = min_idxs
            sr_down = np.ones_like(sc_down) * (data.shape[0] - 1)
            self._sink_indices = np.array([sr_down, sc_down], dtype="int32")
            seed = self.sub2ind(data.shape, sr_down, sc_down).astype("float64")

        elif sink_mode == "mask":
            # All elements in the mask
            coords = np.where(sink_mask != 0)
            sr_down = coords[0]
            sc_down = coords[1]
            self._sink_indices = np.array([sr_down, sc_down], dtype="int32")
            seed = self.sub2ind(data.shape, sr_down, sc_down).astype("float64")

        seed = np.unique(seed)
        seeds = np.concatenate((seeds, seed))

        # Label 2
        label = np.ones_like(seed) * 2
        labels = np.concatenate((labels, label))

        return seeds, labels

    def normalize(self, inp: NDArray) -> NDArray:
        """Normalize an array to [0, 1]"""
        normalized_array: NDArray = (inp - np.min(inp)) / (np.ptp(inp) + self.eps)
        return normalized_array

    def attenuation_weighting(self, img: NDArray, alpha: float) -> NDArray:
        """Compute attenuation weighting

        Args:
            img (NDArray): Image
            alpha: Attenuation coefficient (see publication)

        Returns:
            w (NDArray): Weighting expressing depth-dependent attenuation
        """

        # Create depth vector and repeat it for each column
        dw = np.linspace(0, 1, img.shape[0], dtype="float64")
        dw = np.tile(dw.reshape(-1, 1), (1, img.shape[1]))

        w: NDArray = 1.0 - np.exp(-alpha * dw)  # Compute exp inline

        return w

    def confidence_laplacian(self, padded_index: NDArray, padded_image: NDArray, beta: float, gamma: float):
        """Compute 6-Connected Laplacian for confidence estimation problem

        Args:
            padded_index (NDArray): The index matrix of the image with boundary padding.
            padded_image (NDArray): The padded image.
            beta (float): Random walks parameter that defines the sensitivity of the Gaussian weighting function.
            gamma (float): Horizontal penalty factor that adjusts the weight of horizontal edges in the Laplacian.

        Returns:
            L (csc_matrix): The 6-connected Laplacian matrix used for confidence map estimation.
        """

        m, _ = padded_index.shape

        padded_index = padded_index.T.flatten()
        padded_image = padded_image.T.flatten()

        p = np.where(padded_index > 0)[0]

        i = padded_index[p] - 1  # Index vector
        j = padded_index[p] - 1  # Index vector
        # Entries vector, initially for diagonal
        s = np.zeros_like(p, dtype="float64")

        edge_templates = [
            -1,  # Vertical edges
            1,
            m - 1,  # Diagonal edges
            m + 1,
            -m - 1,
            -m + 1,
            m,  # Horizontal edges
            -m,
        ]

        vertical_end = None

        for iter_idx, k in enumerate(edge_templates):
            neigh_idxs = padded_index[p + k]

            q = np.where(neigh_idxs > 0)[0]

            ii = padded_index[p[q]] - 1
            i = np.concatenate((i, ii))
            jj = neigh_idxs[q] - 1
            j = np.concatenate((j, jj))
            w = np.abs(padded_image[p[ii]] - padded_image[p[jj]])  # Intensity derived weight
            s = np.concatenate((s, w))

            if iter_idx == 1:
                vertical_end = s.shape[0]  # Vertical edges length
            elif iter_idx == 5:
                s.shape[0]  # Diagonal edges length

        # Normalize weights
        s = self.normalize(s)

        # Horizontal penalty
        s[vertical_end:] += gamma
        # Here there is a difference between the official MATLAB code and the paper
        # on the edge penalty. We directly implement what the official code does.

        # Normalize differences
        s = self.normalize(s)

        # Gaussian weighting function
        s = -(
            (np.exp(-beta * s, dtype="float64")) + 1e-5
        )  # --> This epsilon changes results drastically default: 10e-6
        # Please notice that it is not 1e-6, it is 10e-6 which is actually different.

        # Create Laplacian, diagonal missing
        lap = csc_matrix((s, (i, j)))

        # Reset diagonal weights to zero for summing
        # up the weighted edge degree in the next step
        lap.setdiag(0)

        # Weighted edge degree
        diag = np.abs(lap.sum(axis=0).A)[0]

        # Finalize Laplacian by completing the diagonal
        lap.setdiag(diag)

        return lap

    def _solve_linear_system(self, lap, rhs):

        if self.use_cg:
            lap_sparse = lap.tocsr()
            ml = ruge_stuben_solver(lap_sparse, coarse_solver="pinv")
            m = ml.aspreconditioner(cycle="V")
            x, _ = cg(lap, rhs, tol=self.cg_tol, maxiter=self.cg_maxiter, M=m)
        else:
            x = spsolve(lap, rhs)

        return x

    def confidence_estimation(self, img, seeds, labels, beta, gamma):
        """Compute confidence map

        Args:
            img (NDArray): Processed image.
            seeds (NDArray): Seeds for the random walks framework. These are indices of the source and sink nodes.
            labels (NDArray): Labels for the random walks framework. These represent the classes or groups of the seeds.
            beta: Random walks parameter that defines the sensitivity of the Gaussian weighting function.
            gamma: Horizontal penalty factor that adjusts the weight of horizontal edges in the Laplacian.

        Returns:
            map: Confidence map which shows the probability of each pixel belonging to the source or sink group.
        """

        # Index matrix with boundary padding
        idx = np.arange(1, img.shape[0] * img.shape[1] + 1).reshape(img.shape[1], img.shape[0]).T
        pad = 1

        padded_idx = np.pad(idx, (pad, pad), "constant", constant_values=(0, 0))
        padded_img = np.pad(img, (pad, pad), "constant", constant_values=(0, 0))

        # Laplacian
        lap = self.confidence_laplacian(padded_idx, padded_img, beta, gamma)

        # Select marked columns from Laplacian to create L_M and B^T
        b = lap[:, seeds]

        # Select marked nodes to create B^T
        n = np.sum(padded_idx > 0).item()
        i_u = np.setdiff1d(np.arange(n), seeds.astype(int))  # Index of unmarked nodes
        b = b[i_u, :]

        # Remove marked nodes from Laplacian by deleting rows and cols
        keep_indices = np.setdiff1d(np.arange(lap.shape[0]), seeds)
        lap = csc_matrix(lap[keep_indices, :][:, keep_indices])

        # Define M matrix
        m = np.zeros((seeds.shape[0], 1), dtype="float64")
        m[:, 0] = labels == 1

        # Right-handside (-B^T*M)
        rhs = -b @ m

        # Solve linear system
        x = self._solve_linear_system(lap, rhs)

        # Prepare output
        probabilities = np.zeros((n,), dtype="float64")
        # Probabilities for unmarked nodes
        probabilities[i_u] = x
        # Max probability for marked node
        probabilities[seeds[labels == 1].astype(int)] = 1.0

        # Final reshape with same size as input image (no padding)
        probabilities = probabilities.reshape((img.shape[1], img.shape[0])).T

        return probabilities

    def __call__(self, data: NDArray, sink_mask: NDArray | None = None) -> NDArray:
        """Compute the confidence map

        Args:
            data (NDArray): RF ultrasound data (one scanline per column) [H x W] 2D array

        Returns:
            map (NDArray): Confidence map [H x W] 2D array
        """

        # Normalize data
        data = data.astype("float64")
        data = self.normalize(data)

        if self.mode == "RF":
            # MATLAB hilbert applies the Hilbert transform to columns
            data = np.abs(hilbert(data, axis=0)).astype("float64")

        seeds, labels = self.get_seed_and_labels(data, self.sink_mode, sink_mask)

        # Attenuation with Beer-Lambert
        w = self.attenuation_weighting(data, self.alpha)

        # Apply weighting directly to image
        # Same as applying it individually during the formation of the
        # Laplacian
        data = data * w

        # Find condidence values
        map_: NDArray = self.confidence_estimation(data, seeds, labels, self.beta, self.gamma)

        return map_

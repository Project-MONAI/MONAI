# Copyright 2020 MONAI Consortium
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
import torch

from monai.networks.layers.permutohedrallattice import PermutohedralLattice as pl

class CRF(torch.nn.Module):
    def __init__(self, num_iter=5, num_classes=4,
                    alpha=5., beta=5., gamma=5., w_1=1, w_2=1,
                    class_distance_matrix=None,
                ):
        """
        Conditional random field implemeneted using the mean field approximation, see [1].

        Refferences:
        [1] Conditional Random Fields as Recurrent Neural Networks https://arxiv.org/abs/1502.03240

        :param num_iter: number of iterations.
        :param num_classes: number of classes.
        :param alpha: bandwidth for spatial coordinates in bilateral kernel.
            Higher values cause more spatial blurring.
        :param beta: bandwidth for feature coordinates in bilateral kernel.
            Higher values cause more feature blurring.
        :param gamma: bandwidth for spatial coordinates in spatial kernel
            Higher values cause more spatial blurring.
        :param w_1: weight of the bilateral kernel.
        :param w_2: weight of the spatial kernel.
        :param class_distance_matrix: (optional) distance matrix between the classes used as
            compatibility matrix. By default the Potts model is used
            (0s on the diagonal and 1s everywhere else).
        """
        super(CRF, self).__init__()

        self.num_iter = num_iter
        self.theta_alpha = alpha
        self.theta_beta = beta
        self.theta_gamma = gamma
        self.w_1 = w_1
        self.w_2 = w_2

        # using purmutohedral optimised filtering
        self.message_passing_filtering = pl.apply

        # setup class compatibility matrix using the Potts model by default
        if class_distance_matrix is None:
            self.compatibility_weights = self.init_weights_Potts_model(num_classes)
        else:
            if isinstance(class_distance_matrix, np.array):
                class_distance_matrix = torch.from_numpy(class_distance_matrix)
            self.compatibility_weights = class_distance_matrix

    def forward(self, log_unary, features_pairwise):
        """
        Implementation of Algorithm 1 in [1].
        Performs approximated inference for a fully-connected 3D CRF
        with a fixed number of iterations.
        You can see this as a fancy softmax layer.
        Ref:
        [1] "Efficient Inference in Fully Connected CRFs
            with Gaussian Edge Potentials",
            P. Krahenbuhl et al, NIPS 2011.
        :param log_unary: in [1] your classes scores map (output of your CNN before softmax)
        :param features_pairwise: in [1] your input image (can be multi-modal)
        :return: segmentation probabilities map
        """
        # Initialize Q and flatten the spatial dimensions
        batch_size, num_chan, x_dim, y_dim, z_dim = log_unary.size()
        num_vox = x_dim * y_dim * z_dim
        flatten_log_unary = torch.reshape(log_unary, (batch_size, num_chan, -1))
        q = torch.nn.functional.softmax(flatten_log_unary, dim=1)  # seg proba map for the mean field

        # Create the spatial coordinates map
        spatial_x, spatial_y, spatial_z = torch.meshgrid(
            torch.arange(x_dim).cuda(),
            torch.arange(y_dim).cuda(),
            torch.arange(z_dim).cuda()
        )
        spatial = torch.stack([spatial_x, spatial_y, spatial_z], dim=0)  # 4d tensor
        # Duplicate the coordinates along the batch dimension
        spatial = spatial.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # 5d tensor
        spatial = spatial.type(torch.cuda.FloatTensor).detach()
        spatial = torch.reshape(spatial, (batch_size, spatial.size(1), -1))

        # Create the bilateral kernel features
        # Features for the first term of eq (3) in [1]
        img_fea = torch.reshape(features_pairwise, (batch_size, features_pairwise.size(1), -1))
        features_1 = torch.cat([spatial / self.theta_alpha, img_fea / self.theta_beta], dim=1)

        # Create the spatial kernel
        # Features for the second term of eq (3) in [1]
        features_2 = spatial / self.theta_gamma

        # Compute the kernel norm maps to normalize the kernels as in NiftyNet implementation
        ones = torch.ones((batch_size, 1, num_vox)).cuda()
        norm_1 = self.message_passing_filtering(features_1, ones)
        scale_1 = torch.rsqrt(norm_1)  # 1 / sqrt(norm_1)
        norm_2 = self.message_passing_filtering(features_2, ones)
        scale_2 = torch.rsqrt(norm_2)

        # Mean field loop
        for _ in range(self.num_iter):
            # Message passing for each of the two kernels
            q_1 = q * scale_1
            q_1 = self.message_passing_filtering(features_1, q_1)
            q_1 *= scale_1
            q_2 = q * scale_2
            q_2 = self.message_passing_filtering(features_2, q_2)
            q_2 *= scale_2
            # Compatibility update (Potts model only for now)
            q_update = self.compatibility(q_1, q_2)
            # Local update and normalize
            q = torch.nn.functional.softmax(flatten_log_unary - q_update, dim=1)

        # Reshape to initial shape
        q = torch.reshape(q, (batch_size, num_chan, x_dim, y_dim, z_dim))
        return q

    def compatibility(self, q_1, q_2):
        q_comb = self.w_1 * q_1 + self.w_2 * q_2
        q_update = torch.nn.functional.conv1d(q_comb, weight=self.compatibility_weights, bias=None)
        return q_update

    def init_weights_Potts_model(self, num_classes):
        w = np.ones((num_classes, num_classes, 1)).astype(np.float32)
        diag = np.eye(num_classes).reshape((num_classes, num_classes, 1)).astype(np.float32)
        w = w - diag
        weights = torch.from_numpy(w).cuda()
        return weights

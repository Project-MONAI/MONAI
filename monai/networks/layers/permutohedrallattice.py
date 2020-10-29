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

from monai.datastructures.hashtable import HashTable

class PermutohedralLattice(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, feat, desc):
        rank, barycentric, blur_neighbours1, blur_neighbours2, indices = PermutohedralLattice.prepare(feat)
        splat, sliced = PermutohedralLattice.permutohedral_compute(desc,
                          barycentric,
                          blur_neighbours1,
                          blur_neighbours2,
                          indices)
        ctx.save_for_backward(desc)
        ctx.splat = splat
        ctx.rank = rank 
        ctx.barycentric = barycentric
        ctx.bl_ne1 = blur_neighbours1
        ctx.bl_ne2 = blur_neighbours2
        ctx.indices = indices
        return sliced
        
    @staticmethod
    def backward(ctx, grad_output):
        desc, = ctx.saved_tensors
        splat = ctx.splat
        rank = ctx.rank
        barycentric = ctx.barycentric
        bl_ne1 = ctx.bl_ne1
        bl_ne2 = ctx.bl_ne2
        indices = ctx.indices
        out = PermutohedralLattice.permutohedral_compute_gradient(grad_output,
                                                                  desc,
                                                                  splat,
                                                                  rank,
                                                                  barycentric,
                                                                  bl_ne1,
                                                                  bl_ne2,
                                                                  indices)
        return out
    
    def prepare(feat):
        B, n_ch, n_voxels = feat.size()
        n_ch_1 = n_ch + 1
        conv_tensor = np.tril(np.ones((n_ch+1, n_ch+1)), -1).T
        conv_tensor += np.diag([-i for i in range(n_ch+1)])
        conv_tensor = conv_tensor[:, 1:]
        conv_tensor = np.matmul(conv_tensor, 
                                np.sqrt(np.diag([1/(d*(d+1)) for d in range(1, n_ch+1)]))
                               )
        inv_std_dev = np.sqrt(2 / 3.) * (n_ch + 1)
        conv_tensor *= inv_std_dev
        conv_tensor = conv_tensor[:, :, np.newaxis]
        feat = torch.nn.functional.conv1d(feat, torch.FloatTensor(conv_tensor).cuda())
    
        feat_v = torch.round(feat / (n_ch + 1))
        rem0 = feat_v * (n_ch + 1)
        index = torch.argsort(feat - rem0, dim=1, descending=True)
        rank = torch.argsort(index, dim=1, descending=False)
        rank = rank + torch.sum(feat_v, 1).unsqueeze(1).type(torch.cuda.LongTensor)
        add_minus = (rank<0).type(torch.cuda.LongTensor) - (rank>n_ch).type(torch.cuda.LongTensor)
        add_minus *= (n_ch + 1)
        rank = rank + add_minus
        rem0 = rem0 + add_minus.type(torch.cuda.FloatTensor)
        
        y = (feat - rem0) / (n_ch + 1)
        v_sorted = torch.sort(y, dim = 1, descending=False)[0]
        barycentric = v_sorted - torch.cat([v_sorted[:, -1:] - 1., v_sorted[:, :-1]], 1)

        canonical = torch.cuda.FloatTensor(
        [[i] * ((n_ch + 1) - i) + [i - (n_ch + 1)] * i for i in range(n_ch + 1)])
        
        def _simple_hash(key):
            key = key.type(torch.cuda.DoubleTensor)
            hash_vector = np.floor(np.power(np.iinfo(np.int64).max, 1. / (n_ch + 2)))
            hash_vector = torch.pow(hash_vector, torch.arange(1, n_ch + 1))
            hash_vector = hash_vector.type(torch.DoubleTensor).unsqueeze(0)
            hash_vector = hash_vector.cuda()
            if len(key.size())==3:
                hash_vector = hash_vector.unsqueeze(2)
                return torch.sum(key * hash_vector.repeat(key.size(0), 1, key.size(-1)), 1)
            if len(key.size())==2:
                return torch.sum(key * hash_vector.repeat(key.size(0), 1), 1)

        dic_hash_lattice = HashTable(n_ch, torch.cuda.DoubleTensor, 2**30)
        loc = [None] * (n_ch + 1)
        loc_hash = [None] * (n_ch + 1)
        for scit in range(n_ch + 1):
            loc[scit] = torch.gather(canonical[scit:scit+1].unsqueeze(2).repeat(rank.size(0), 1, rank.size(2)), 
                                     1, 
                                     rank[:, :-1])
            loc[scit] += rem0[:, :-1]
            loc[scit] = loc[scit]
            loc_hash[scit] = _simple_hash(loc[scit])
            loc[scit] = torch.reshape(loc[scit].permute(0, 2, 1), (-1, n_ch))
            dic_hash_lattice.add_values(loc_hash[scit].view(-1), loc[scit])
        
        dic_hash_lattice.filter_values()
        fused_loc = dic_hash_lattice.export_values()
        dic_hash_lattice.update_rank()

        indices = [None] * n_ch_1
        blur_neighbours1 = [None] * n_ch_1
        blur_neighbours2 = [None] * n_ch_1
        default = torch.tensor(0).type(torch.LongTensor).cuda()
        for dit in range(n_ch_1):
            offset = [n_ch if i == dit else -1 for i in range(n_ch)]
            offset = torch.cuda.FloatTensor(offset) 
            blur_neighbours1[dit] = dic_hash_lattice.get_rank(_simple_hash(fused_loc + offset).view(-1))[:, 0]
            blur_neighbours2[dit] = dic_hash_lattice.get_rank(_simple_hash(fused_loc - offset).view(-1))[:, 0]
            indices[dit] = dic_hash_lattice.get_rank(loc_hash[dit].view(-1)).view(B, n_voxels)
        return rank, barycentric, blur_neighbours1, blur_neighbours2, indices
    
    def permutohedral_compute(data_vector, barycentric, blur_neighbours1, blur_neighbours2, indices):
        n_ch_1 = barycentric.size(1)
        n_ch = n_ch_1 - 1
        B, n_ch_data, n_voxels = data_vector.size()

        # Splatting
        splat = torch.zeros((B, n_ch_data, blur_neighbours1[0].size(0) + 1)).cuda()

        for scit in range(n_ch_1):
            simplex_indices = indices[scit].unsqueeze(1).repeat(1, data_vector.size(1), 1)
            bcWeights = barycentric[:, scit:scit+1].repeat(1, data_vector.size(1), 1)

            splat.scatter_add_(2, simplex_indices, bcWeights * data_vector)
            
        
        # Blur with 1D kernels
        for dit in range(n_ch + 1):
            b1 = torch.index_select(splat, 2, blur_neighbours1[dit])
            b3 = torch.index_select(splat, 2, blur_neighbours2[dit])
            splat = torch.cat([splat[:, :, :1], splat[:, :, 1:] + 0.5 * (b1 + b3)], 2)

        # Slice
        sliced = 0.0
        alpha = 1. / (1. + np.power(2., -n_ch))

        for scit in range(0, n_ch_1):
            simplex_indices = indices[scit].unsqueeze(1).repeat(1, splat.size(1), 1)
            bcweight = barycentric[:, scit:scit+1].repeat(1, splat.size(1), 1)
            
            sliced += alpha * bcweight * torch.gather(splat, 2, simplex_indices)
            
        return splat, sliced
    
        
    def permutohedral_compute_gradient(data_vector,
                          data_vector_real,
                          blured,
                          rank, 
                          barycentric,
                          blur_neighbours1,
                          blur_neighbours2,
                          indices,
                          low_precision=False):
        n_ch_1 = barycentric.size(1)
        n_ch = n_ch_1 - 1
        n_ch_1 = n_ch + 1
        
        B, n_ch_data, n_voxels = data_vector.size()

        # Splatting
        splat = torch.zeros((B, n_ch_data, blur_neighbours1[0].size(0)+1)).cuda()
        for scit in range(n_ch_1):
            data = (data_vector * 
                    barycentric[:, scit:scit+1].repeat(1, data_vector.size(1), 1))
            splat.scatter_add_(2, 
                               indices[scit].unsqueeze(1).repeat(1, data.size(1), 1), 
                               data)

        # Blur with 1D kernels
        for dit in range(n_ch, -1, -1):
            b1 = torch.index_select(splat, 2, blur_neighbours1[dit])
            b3 = torch.index_select(splat, 2, blur_neighbours2[dit])
            splat = torch.cat([
                splat[:, :, :1], splat[:, :, 1:] + 0.5 * (b1 + b3)], 2)
            
        # Slice
        sliced_feat = [None] * n_ch_1
        alpha = 1. / (1. + np.power(2., -n_ch))
        for scit in range(0, n_ch_1):
            grads = torch.gather(splat, 
                                 2, 
                                 indices[scit].unsqueeze(1).repeat(1, splat.size(1), 1))
            sliced_feat[scit] = (grads * 
                                 data_vector_real * 
                                 alpha)
            sliced_feat[scit] = sliced_feat[scit].sum(1)
        sliced_feat = torch.stack(sliced_feat, 1)
        
        sliced_feat_bis = [None] * n_ch_1
        for dim in range(0, n_ch_1):
            a = torch.gather(blured, 
                             2, 
                             indices[dim].unsqueeze(1).repeat(1, blured.size(1), 1))
            sliced_feat_bis[dim] = (a * data_vector * alpha)
            sliced_feat_bis[dim] = sliced_feat_bis[dim].sum(1)
        sliced_feat_bis = torch.stack(sliced_feat_bis, 1)

        sliced_feat += sliced_feat_bis
        
        ### derivative w.r.t barycentric coordinates ###

        conv_b = np.zeros((n_ch+1, n_ch+1))
        conv_b[0, 0] = -1
        conv_b[0, n_ch] = 1
        for i in range(1, n_ch+1):
            conv_b[i, n_ch-i] = 1
            conv_b[i, n_ch-i+1] = -1
        conv_b = conv_b.T
        conv_b = torch.cuda.FloatTensor(conv_b / (n_ch + 1)).unsqueeze(2)
        sliced_feat = F.conv1d(sliced_feat, conv_b)
        
        ### derivative w.r.t y (features in canonical simplex) ###
        sliced_feat = torch.gather(sliced_feat, 1, rank)

        ### derivative w.r.t embedded (in lattice space) u ###

        conv_tensor = np.tril(np.ones((n_ch+1, n_ch+1)), -1).T
        conv_tensor += np.diag([-i for i in range(n_ch+1)])
        conv_tensor = conv_tensor[:, 1:]
        conv_tensor = np.matmul(conv_tensor, np.sqrt(np.diag([1/(d*(d+1)) for d in range(1, n_ch+1)])))
        conv_tensor = np.expand_dims(conv_tensor.T, 2)
        inv_std_dev = np.sqrt(2 / 3.) * (n_ch + 1)
        conv_tensor *= inv_std_dev
        conv_filter = torch.cuda.FloatTensor(conv_tensor)

        ### derivative w.r.t u ###

        sliced_feat = F.conv1d(sliced_feat, conv_filter)

        sliced_desc = 0.0
        alpha = 1. / (1. + np.power(2., -n_ch))
        for scit in range(0, n_ch_1):
            simplex_indices = indices[scit].unsqueeze(1).repeat(1, splat.size(1), 1)
            bcweight = barycentric[:, scit:scit+1].repeat(1, splat.size(1), 1)

            sliced_desc += alpha * bcweight * torch.gather(splat, 2, simplex_indices)
        
        return sliced_feat, sliced_desc  
        
        
        
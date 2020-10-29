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

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

from monai.utils import optional_import

_C, _ = optional_import("monai._C")

def _simple_hash(key, hash_vector, table_size):
    res = (key*hash_vector).sum(dim=1)
    return (res%table_size).type(torch.cuda.IntTensor)

class PermutohedralLattice(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, feat, desc):
        """
        feat : B, D_f, N
        desc : B, D_d, N
        """
        rank, barycentric, blur_neighbours1, blur_neighbours2, indices = PermutohedralLattice.prepare(feat)
        splat, sliced = PermutohedralLattice.permutohedral_compute(desc,
                          barycentric,
                          blur_neighbours1,
                          blur_neighbours2,
                          indices)
        ctx.desc = desc
        ctx.splat = splat
        ctx.rank = rank 
        ctx.barycentric = barycentric
        ctx.bl_ne1 = blur_neighbours1
        ctx.bl_ne2 = blur_neighbours2
        ctx.indices = indices
        return sliced
        
    @staticmethod
    def backward(ctx, grad_output):
        desc = ctx.desc
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
        del(ctx.bl_ne1)
        del(ctx.bl_ne2)
        del(ctx.indices)
        del(ctx.barycentric)
        del(ctx.rank)
        del(ctx.splat)
        del(ctx.desc)
        return out
    
    @staticmethod
    def prepare(feat):
        B, n_ch, n_voxels = feat.size()
        ### embed features into higher dimension space ###
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
        feat = F.conv1d(feat, torch.FloatTensor(conv_tensor).cuda())
    
        feat_v = torch.round(feat / (n_ch + 1))
        rem0 = feat_v * (n_ch + 1)
        index = torch.argsort(feat - rem0, dim=1, descending=True)
        rank = torch.argsort(index, dim=1, descending=False)
        rank = rank + torch.sum(feat_v, 1).unsqueeze(1).type(torch.cuda.LongTensor)
        add_minus = (rank<0).type(torch.cuda.LongTensor) - (rank>n_ch).type(torch.cuda.LongTensor)
        add_minus *= (n_ch + 1)
        rank = rank + add_minus
        rem0 = rem0.type(torch.cuda.LongTensor) + add_minus
        
        y = (feat - rem0.type(torch.cuda.FloatTensor)) / (n_ch + 1)
        v_sorted = torch.sort(y, dim = 1, descending=False)[0]
        barycentric = v_sorted - torch.cat([v_sorted[:, -1:] - 1., 
                                         v_sorted[:, :-1]], 1)
        canonical = torch.LongTensor(
        [[i] * ((n_ch + 1) - i) + [i - (n_ch + 1)] * i for i in range(n_ch + 1)]).cuda()

        rem0 = rem0.permute(0, 2, 1).reshape((-1, n_ch_1))
        rem0 = rem0.contiguous()
        rank = rank.permute(0, 2, 1).reshape((-1, n_ch_1))
        rank = rank.contiguous()

        table_size = 20000003
        hash_vector = int(np.floor(np.power(table_size, 1. / (n_ch))))
        hash_vector = torch.pow(hash_vector, torch.arange(0, n_ch))
        hash_vector = hash_vector.type(torch.IntTensor)
        hash_vector = hash_vector.cuda()

        table = torch.zeros((table_size, n_ch_1)).type(torch.IntTensor).cuda() - 2
        n_entries = torch.zeros((1,)).type(torch.IntTensor).cuda() + 1

        loc = [None] * (n_ch + 1)
        loc_hash = [None] * (n_ch + 1)
        for scit in range(n_ch + 1):
            loc[scit] = torch.gather(canonical[scit:scit+1].repeat(rank.size(0), 1), 
                                        1, 
                                        rank[:, :-1])
            loc[scit] += rem0[:, :-1]
            loc_hash[scit] = _simple_hash(loc[scit], hash_vector, table_size)
            _ = _C.hashtable_insert(table, n_entries, loc[scit].type(torch.cuda.IntTensor), loc_hash[scit].type(torch.cuda.IntTensor))
            del(_)
        
        fused_loc = _C.hashtable_get_values(table, n_entries[0].item())

        indices = [None] * n_ch_1
        blur_neighbours1 = [None] * n_ch_1
        blur_neighbours2 = [None] * n_ch_1
        for dit in range(n_ch_1):
            offset = [n_ch if i == dit else -1 for i in range(n_ch)]
            offset = torch.cuda.IntTensor(offset) 
            blur_neighbours1[dit] = _C.hashtable_get_rank(table, 
                                                    fused_loc + offset, 
                                                    _simple_hash(fused_loc + offset, hash_vector, table_size))
            blur_neighbours2[dit] = _C.hashtable_get_rank(table, 
                                                    fused_loc - offset, 
                                                    _simple_hash(fused_loc - offset, hash_vector, table_size))
            indices[dit] = _C.hashtable_get_rank(table.type(torch.cuda.IntTensor),
                                        loc[dit].type(torch.cuda.IntTensor),
                                        loc_hash[dit].type(torch.cuda.IntTensor))
        indices = torch.stack(indices, dim=1)
        N_P = torch.stack(blur_neighbours1, dim=1)
        N_N = torch.stack(blur_neighbours2, dim=1)

        # indices, N_P, N_N = latticehtopp_nlfa.latticehtopp(canonical, rem0, rank)

        indices = indices.reshape((B, n_voxels, n_ch_1)).permute(0, 2, 1)
        rank = rank.reshape((B, n_voxels, n_ch_1)).permute(0, 2, 1)

        return rank, barycentric, N_P, N_N, indices
    
    @staticmethod
    def permutohedral_compute(data_vector,
                          barycentric,
                          blur_neighbours1,
                          blur_neighbours2,
                          indices):
        
        indices = indices.type(torch.cuda.LongTensor)
        blur_neighbours1 = blur_neighbours1.type(torch.cuda.LongTensor)
        blur_neighbours2 = blur_neighbours2.type(torch.cuda.LongTensor)

        n_ch_1 = barycentric.size(1)
        n_ch = n_ch_1 - 1
        B, n_ch_data, n_voxels = data_vector.size()
        # Splatting
        splat = torch.zeros((B, n_ch_data, blur_neighbours1.size(0) + 1)).cuda()
        for scit in range(n_ch_1):
            data = (data_vector * 
                    barycentric[:, scit:scit+1].repeat(1, data_vector.size(1), 1))
            splat.scatter_add_(2, 
                               indices[:, scit:scit+1].repeat(1, data.size(1), 1), 
                               data)
        # print(splat)
        # print(splat.size())
        # caca

        # Blur with 1D kernels
        for dit in range(n_ch + 1):
            b1 = torch.index_select(splat, 2, blur_neighbours1[:, dit])
            b3 = torch.index_select(splat, 2, blur_neighbours2[:, dit])
            splat = torch.cat([
                splat[:, :, :1], splat[:, :, 1:] + 0.5 * (b1 + b3)], 2)
        # Slice
        sliced = 0.0
        # Alpha is a magic scaling constant from CRFAsRNN code
        alpha = 1. / (1. + np.power(2., -n_ch))
        for scit in range(0, n_ch_1):
            sliced += (torch.gather(splat, 
                                   2, 
                                   indices[:, scit:scit+1].repeat(1, splat.size(1), 1)
                                  ) * 
            barycentric[:, scit:scit+1].repeat(1, splat.size(1), 1) *  
            alpha)
            
        return splat, sliced
    
    @staticmethod    
    def permutohedral_compute_gradient(data_vector,
                          data_vector_real,
                          blured,
                          rank, 
                          barycentric,
                          blur_neighbours1,
                          blur_neighbours2,
                          indices):
        
        indices = indices.type(torch.cuda.LongTensor)
        blur_neighbours1 = blur_neighbours1.type(torch.cuda.LongTensor)
        blur_neighbours2 = blur_neighbours2.type(torch.cuda.LongTensor)
        
        n_ch_1 = barycentric.size(1)
        n_ch = n_ch_1 - 1
        
        B, n_ch_data, n_voxels = data_vector.size()

        alpha = 1. / (1. + np.power(2., -n_ch))

        ### derivative w.r.t barycentric coordinates of loc ###
        splat = torch.zeros((B, n_ch_data, blur_neighbours1.size(0)+1)).cuda()
        for scit in range(n_ch_1):
            data = (data_vector * 
                    barycentric[:, scit:scit+1].repeat(1, data_vector.size(1), 1) * alpha)
            splat.scatter_add_(2, 
                               indices[:, scit].unsqueeze(1).repeat(1, data.size(1), 1), 
                               data)

        for dit in range(n_ch, -1, -1):
            b1 = torch.index_select(splat, 2, blur_neighbours1[:, dit])
            b3 = torch.index_select(splat, 2, blur_neighbours2[:, dit])
            splat = torch.cat([
                splat[:, :, :1], splat[:, :, 1:] + 0.5 * (b1 + b3)], 2)
            
        sliced_loc = [None] * n_ch_1
        for scit in range(0, n_ch_1):
            grads = torch.gather(splat, 
                                 2, 
                                 indices[:, scit].unsqueeze(1).repeat(1, splat.size(1), 1))
            sliced_loc[scit] = (grads * 
                                 data_vector_real)
            sliced_loc[scit] = sliced_loc[scit].sum(1)
        sliced_loc = torch.stack(sliced_loc, 1)

        ### derivative w.r.t barycentric coordinates of dest ###
        sliced_dest = [None] * n_ch_1
        for dim in range(0, n_ch_1):
            a = torch.gather(blured, 
                             2, 
                             indices[:, dim].unsqueeze(1).repeat(1, blured.size(1), 1))
            sliced_dest[dim] = (a * data_vector * alpha)
            sliced_dest[dim] = sliced_dest[dim].sum(1)
        sliced_dest = torch.stack(sliced_dest, 1)

        ### derivative w.r.t y (features in canonical simplex) ###
        conv_b = np.zeros((n_ch+1, n_ch+1))
        conv_b[0, 0] = -1
        conv_b[0, n_ch] = 1
        for i in range(1, n_ch+1):
            conv_b[i, n_ch-i] = 1
            conv_b[i, n_ch-i+1] = -1
        conv_b = conv_b.T
        conv_b = torch.cuda.FloatTensor(conv_b / (n_ch + 1)).unsqueeze(2)

        sliced_loc = F.conv1d(sliced_loc, conv_b)
        sliced_dest = F.conv1d(sliced_dest, conv_b)
        
        ### derivative w.r.t embedded (in lattice space) u ###
        sliced_loc = torch.gather(sliced_loc, 1, rank)
        sliced_dest = torch.gather(sliced_dest, 1, rank)

        ### derivative w.r.t u ###
        conv_tensor = np.tril(np.ones((n_ch+1, n_ch+1)), -1).T
        conv_tensor += np.diag([-i for i in range(n_ch+1)])
        conv_tensor = conv_tensor[:, 1:]
        conv_tensor = np.matmul(conv_tensor, np.sqrt(np.diag([1/(d*(d+1)) for d in range(1, n_ch+1)])))
        conv_tensor = np.expand_dims(conv_tensor.T, 2)
        inv_std_dev = np.sqrt(2 / 3.) * (n_ch + 1)
        conv_tensor *= inv_std_dev
        conv_filter = torch.cuda.FloatTensor(conv_tensor)

        sliced_dest = F.conv1d(sliced_dest, conv_filter)
        sliced_loc = F.conv1d(sliced_loc, conv_filter)

        ### derivative w.r.t desc ###
        sliced_desc = 0.0
        for scit in range(0, n_ch_1):
            sliced_desc = sliced_desc + (torch.gather(splat, 
                                         2, 
                                         indices[:, scit].unsqueeze(1).repeat(1, 
                                                                           splat.size(1),
                                                                           1)
                                        ) * 
               barycentric[:, scit:scit+1].repeat(1, splat.size(1), 1))
        
        return sliced_loc + sliced_dest, sliced_desc  
        


























        
if __name__=="__main__":

    # np.random.seed(0)
    # feat1 = np.random.rand(1, 5, 15)
    # feat2 = np.random.rand(1, 5, 7)
    # desc = np.ones((1, 20, 15))
    # pl = PermutohedralLattice.apply
    # feat1 = torch.cuda.FloatTensor(feat1)
    # feat2 = torch.cuda.FloatTensor(feat2)
    # desc = torch.cuda.FloatTensor(desc)
    # feat1.requires_grad = True
    # feat2.requires_grad = True
    # desc.requires_grad = True
    # out = pl(feat1, feat2, desc)
    # print(out)
    # out_ = pl(feat1, feat2, desc)
    # print(out_)
    # add = torch.zeros((1, 20, 15)).cuda()
    # add[0, 0, :] = 0.0001
    # print(pl(feat1, feat2, desc+add))
    # add[0, 0, :] = 0
    # add[0, 1, :] = 0.0001
    # print(pl(feat1, feat2, desc+add))
    # loss = out.sum([0, 1, 2])
    # loss.backward()




    np.random.seed(0)
    feat1 = np.random.rand(1, 5, 15)
    feat2 = np.random.rand(1, 5, 7)
    desc = np.random.rand(1, 10, 15)
    pl = PermutohedralLattice.apply
    feat1 = torch.cuda.FloatTensor(feat1)
    feat2 = torch.cuda.FloatTensor(feat2)
    desc = torch.cuda.FloatTensor(desc)
    feat1.requires_grad = True
    feat2.requires_grad = True
    desc.requires_grad = True
    out = pl(feat1, feat2, desc)
    loss = out.sum([0, 1, 2])
    loss.backward()

    # check_feat1 = torch.zeros(feat1.size()).cuda()
    # for i in range(5):
    #     for j in range(15):
    #         add = torch.zeros(feat1.size()).cuda()
    #         add[0, i, j] = 0.0001
    #         check_feat1[0, i, j] = (pl(feat1+add, feat2, desc) - pl(feat1, feat2, desc)).sum()/0.0001

    # check_feat2 = torch.zeros(feat2.size()).cuda()
    # for i in range(5):
    #     for j in range(7):
    #         add = torch.zeros(feat2.size()).cuda()
    #         add[0, i, j] = 0.0001
    #         check_feat2[0, i, j] = (pl(feat1, feat2+add, desc) - pl(feat1, feat2, desc)).sum()/0.0001

    check_desc = torch.zeros(desc.size()).cuda()
    for i in range(10):
        for j in range(15):
            add = torch.zeros(desc.size()).cuda()
            add[0, i, j] = 0.0001
            check_desc[0, i, j] = (pl(feat1, feat2, desc+add) - pl(feat1, feat2, desc)).sum()/0.0001
    
    # print(check_feat1)
    # print(feat1.grad)
    # print(check_feat2)
    # print(feat2.grad)
    print(check_desc)
    print(desc.grad)

    # print((check_feat1-feat1.grad)/check_feat1)
    # print((check_feat2-feat2.grad)/check_feat2)
    print((check_desc-desc.grad)/check_desc)
    print(check_desc/desc.grad)


#     from torch.autograd import gradcheck
    
# #     def test(a, b):
# #         return pl(feat1, a, b)
    
#     def test(b):
#         return pl(feat1, feat2, b).sum()
    
#     # gradcheck(test, desc, raise_exception=True) 
#     gradcheck(test, feat1, raise_exception=True)
#     # gradcheck(test, feat2, raise_exception=True)  
    

#     def brute_force(feat1, feat2, desc):
#         res = np.zeros(desc.shape)
#         for b in range(feat1.shape[0]):
#             for i in range(feat1.shape[-1]):
#                 for j in range(feat1.shape[-1]):
#                     res[b, :, i] += np.exp(-((feat1[b, :, i] - feat2[b, :, j])**2).sum()) * desc[b, :, j]
#         return res

#     np.random.seed(0)
#     feat1 = 10*np.random.rand(2, 5, 1000)
#     feat2 = 10*np.random.rand(2, 5, 1000)
#     desc = np.random.rand(2, 12, 1000)

#     res = brute_force(feat1, feat2, desc)

#     pl = PermutohedralLattice.apply
#     feat1 = torch.cuda.FloatTensor(feat1)
#     feat2 = torch.cuda.FloatTensor(feat2)
#     desc = torch.cuda.FloatTensor(desc)
#     out = pl(feat1, feat2, desc)
#     out = out.cpu().numpy()

#     relat_error = np.abs(out-res)/(res+0.00001)
#     print(relat_error.max())
#     print(relat_error.mean())
              
        
        
        
        
        
        
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

class HashTable(object):
    def __init__(self, n_ch, dtype, table_size):
        self.n_ch = n_ch
        self.dtype = dtype
        self.indices = torch.cuda.LongTensor([-12])
        self.values = torch.tensor([[0. for i in range(n_ch)]]).type(torch.cuda.FloatTensor)
        self.rank = None
        self.unikeys = None
        self.table_size = table_size
        
    def add_values(self, keys, values):
        if len(values.size())==1:
            values = values.unsqueeze(1)
        keys = keys%self.table_size
        keys = keys.type(torch.cuda.LongTensor)
        keys_ = torch.cat([values.type(torch.cuda.LongTensor), keys.unsqueeze(1)], 1)
        unikeys = torch.unique(keys_, dim=0)
        self.indices = torch.cat([self.indices, unikeys[:, -1]])
        self.values = torch.cat([self.values, unikeys[:, :-1].type(torch.cuda.FloatTensor)], 0)
        if self.indices[0]==(-12):
            self.indices = self.indices[1:]
            self.values = self.values[1:]
    
    def filter_values(self):
        order = torch.argsort(self.indices)
        self.indices = self.indices[order]
        self.values = self.values[order]
        unikeys, reverse_t = torch.unique(self.indices, return_inverse=True, sorted=True)
        if unikeys.size(0)!=self.indices.size(0):
            index = torch.sparse.LongTensor(reverse_t.unsqueeze(0), torch.ones(reverse_t.size(0)).cuda(), torch.Size([unikeys.size(0),])).to_dense()
            index = index.unsqueeze(1)
            step_size=20000
            f_index = min(step_size, unikeys.size(0))
            curr_index = 0
            conv_size = f_index - curr_index
            conv = torch.ones((conv_size, conv_size), device=torch.device('cuda:0'))
            conv = torch.tril(conv)
            while f_index<unikeys.size(0):
                if conv_size!= f_index - curr_index:
                    conv_size = f_index - curr_index
                    conv = torch.ones((conv_size, conv_size), device=torch.device('cuda:0'))
                    conv = torch.tril(conv)
                index[curr_index:f_index] = torch.mm(conv, index[curr_index:f_index])
                curr_index = f_index - 1
                f_index = min(curr_index + step_size, unikeys.size(0))
            conv_size = f_index - curr_index
            conv = torch.ones((conv_size, conv_size), device=torch.device('cuda:0'))
            conv = torch.tril(conv)
            index[curr_index:f_index] = torch.mm(conv, index[curr_index:f_index])
            index = index - 1
            index = index[:, 0].type(torch.cuda.LongTensor)
            self.values = self.values[index]
            self.indices = self.indices[index]
            
    def update_rank(self):
        self.rank = torch.arange(1, self.indices.size(0) + 1).cuda()
    
    def get_values(self, keys):
        keys = keys%self.table_size
        keys = keys.type(torch.cuda.LongTensor)
        res = torch.zeros((keys.size(0), self.values.size(1))).type(torch.cuda.FloatTensor)
        unikeys = torch.unique(keys)
        for k in unikeys:
            res[keys==k] += self.values[self.indices==k]
        return res
    
    def get_rank(self, keys):
        keys = keys%self.table_size
        keys = keys.type(torch.cuda.LongTensor)
        indices = torch.arange(keys.size(0)).cuda().unsqueeze(0)
        indices = torch.cat([indices, keys.unsqueeze(0).type(torch.cuda.LongTensor)], 0)
        indices_sp_m = torch.sparse.LongTensor(indices, torch.ones(indices.size(1)).type(torch.FloatTensor).cuda(), torch.Size([indices.size(1), self.table_size]))
        index_rank_spm = torch.cat([self.indices.unsqueeze(0), torch.zeros((1, self.indices.size(0))).type(torch.LongTensor).cuda()], 0)
        rank_sp_m = torch.sparse.LongTensor(index_rank_spm, self.rank.type(torch.cuda.FloatTensor), torch.Size([self.table_size, 1])).to_dense()
        prod = torch.sparse.mm(indices_sp_m, rank_sp_m)
        return prod.type(torch.cuda.LongTensor)
    
    def export(self):
        return self.indices, self.values
    
    def export_values(self):
        return self.values
    
    def export_indices(self):
        return self.indices
    
    def clear_table(self):
        del(self.indices)
        del(self.values)
        del(self.rank)
        
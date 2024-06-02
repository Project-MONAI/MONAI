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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import math

def get_gaussian_kernel_2d(ksize=3, sigma=1):
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp( 
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance + 1e-16)
        )
    return gaussian_kernel / torch.sum(gaussian_kernel)

class get_svls_filter_2d(torch.nn.Module):
    def __init__(self, ksize=3, sigma=1, channels=0):
        super(get_svls_filter_2d, self).__init__()
        gkernel = get_gaussian_kernel_2d(ksize=ksize, sigma=sigma)
        neighbors_sum = (1 - gkernel[1,1]) + 1e-16
        gkernel[int(ksize/2), int(ksize/2)] = neighbors_sum
        self.svls_kernel = gkernel / neighbors_sum
        svls_kernel_2d = self.svls_kernel.view(1, 1, ksize, ksize)
        svls_kernel_2d = svls_kernel_2d.repeat(channels, 1, 1, 1)
        padding = int(ksize/2)
        self.svls_layer = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=ksize, groups=channels,
                                    bias=False, padding=padding, padding_mode='replicate')
        self.svls_layer.weight.data = svls_kernel_2d
        self.svls_layer.weight.requires_grad = False
    def forward(self, x):
        return self.svls_layer(x) / self.svls_kernel.sum()

class NACLLoss(_Loss):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)
    """
    def __init__(self,
                 classes=None,
                 kernel_size=3,
                 kernel_ops='mean',
                 distance_type='l1',
                 is_softmax=False,
                 alpha=0.1,
                 ignore_index=-100,
                 sigma=1,
                 schedule=""):
        
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        
        self.distance_type = distance_type
        
        self.alpha = alpha
        self.ignore_index = ignore_index
        
        self.is_softmax = is_softmax

        self.nc = classes
        self.ks = kernel_size
        self.kernel_ops = kernel_ops
        self.cross_entropy = nn.CrossEntropyLoss()
        if kernel_ops == 'gaussian':
            self.svls_layer = get_svls_filter_2d(ksize=kernel_size, sigma=sigma, channels=classes)

    def get_constr_target(self, mask):
        
        mask = mask.unsqueeze(1) ## unfold works for 4d. 
        
        bs, _, h, w = mask.shape
        unfold = torch.nn.Unfold(kernel_size=(self.ks, self.ks),padding=self.ks // 2)    
        
        rmask = []
        
        if self.kernel_ops == 'mean':        
            umask = unfold(mask.float())
                
            for ii in range(self.nc):
                rmask.append(torch.sum(umask == ii,1)/self.ks**2)
                
        if self.kernel_ops == 'gaussian':

            oh_labels = F.one_hot(mask[:,0].to(torch.int64), num_classes = self.nc).contiguous().permute(0,3,1,2).float()
            rmask = self.svls_layer(oh_labels)

            return rmask

        rmask = torch.stack(rmask,dim=1)
        rmask = rmask.reshape(bs, self.nc, h, w)
            
        return rmask
        

    def forward(self, inputs, targets, imgs):
        
        loss_ce = self.cross_entropy(inputs, targets)
        
        utargets = self.get_constr_target(targets, imgs)
        
        if self.is_softmax:
            inputs = F.softmax(inputs, dim=1)
        
        if self.distance_type == 'l1':
            loss_conf = torch.abs(utargets - inputs).mean()  
            
        if self.distance_type == 'l2':    
            loss_conf = (torch.abs(utargets - inputs)**2).mean()  

        loss = loss_ce + self.alpha * loss_conf

        return loss, loss_ce, loss_conf

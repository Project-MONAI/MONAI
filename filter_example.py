import numpy as np
import torch

from monai.networks.layers.filtering import BilateralFilter
from monai.networks.layers.permutohedrallattice import PermutohedralLattice as phl

import skimage
from skimage.restoration import denoise_bilateral as skimage_bilateral
import matplotlib.pyplot as plt
import time

def bilateral_phl(input_tensor, spatial_sigma, color_sigma):
    x_coords, y_coords = torch.meshgrid(torch.arange(input_tensor.size(2)), torch.arange(input_tensor.size(3))) 
    spatial_features = torch.stack([x_coords, y_coords]).unsqueeze(0).type(torch.FloatTensor)
    
    feature_tensor = torch.cat([spatial_features / spatial_sigma, input_tensor / color_sigma], dim=1)

    # flattening spatial dimensions
    input_tensor_shape = input_tensor.size()

    input_tensor = torch.flatten(input_tensor, start_dim=2).cuda()
    feature_tensor = torch.flatten(feature_tensor, start_dim=2).cuda()

    # applying phl twice, once to acquire a normalisation mask, then again with the real data
    normalisation_mask_tensor = phl.apply(feature_tensor, torch.ones_like(input_tensor).cuda())
    filtered_tensor = phl.apply(feature_tensor, input_tensor) / normalisation_mask_tensor

    # reshaping output
    filtered_tensor = filtered_tensor.reshape(input_tensor_shape).cpu()

    return filtered_tensor

# filter parameters
spatial_sigma = 10.0
color_sigma = 10.0

data = np.array(skimage.data.astronaut()) / 255.0
data_tensor = torch.from_numpy(data).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(0)

start = time.time()
cpp_phl_tensor = BilateralFilter.apply(data_tensor, spatial_sigma, color_sigma, True)
cpp_phl_time = time.time() - start
print(cpp_phl_time)

start = time.time()
cpp_std_tensor = BilateralFilter.apply(data_tensor, spatial_sigma, color_sigma, False)
cpp_std_time = time.time() - start
print(cpp_std_time)

start = time.time()
pyt_phl_tensor = bilateral_phl(data_tensor, spatial_sigma, color_sigma)
pyt_phl_time = time.time() - start
print(pyt_phl_time)

cpp_phl_result = cpp_phl_tensor.squeeze(0).permute(1, 2, 0).numpy()
cpp_std_result = cpp_std_tensor.squeeze(0).permute(1, 2, 0).numpy()
pyt_phl_result = pyt_phl_tensor.squeeze(0).permute(1, 2, 0).numpy()

# results
plt.subplot(221).axis('off')
plt.title('input image')
plt.imshow(data)
plt.subplot(222).axis('off')
plt.title('pytorch cuda phl: {}sec'.format('%.2f' % pyt_phl_time))
plt.imshow(pyt_phl_result)
plt.subplot(223).axis('off')
plt.title('c++ cpu simple: {}sec'.format('%.2f' % cpp_std_time))
plt.imshow(cpp_std_result)
plt.subplot(224).axis('off')
plt.title('c++ cpu phl: {}sec'.format('%.2f' % cpp_phl_time))
plt.imshow(cpp_phl_result)
plt.show()

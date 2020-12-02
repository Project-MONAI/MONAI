import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from monai.networks.layers.filtering import BilateralFilter

import nibabel as nib
import matplotlib.pyplot as plt
import time

# helper method for plotting
def show_result(position, data):
    plt.subplot(position).axis('off')
    plt.imshow(data, cmap='gray')

# load ct scan data (128x128x128)
scan = nib.load('temp_abdominal_ct.nii').get_fdata()
scan += np.min(scan)
scan /= np.max(scan)

# data into tensor
scan_tensor = torch.from_numpy(scan).type(torch.FloatTensor).contiguous()
scan_tensor_cuda = scan_tensor.cuda()

# filter parameters
spatial_sigma = 30
color_sigma = 0.05

# filter
print('running 3D bilateral filter...')

print('cpu bruteforce')
start=time.time()
result_tensor_cpu_brute = BilateralFilter.apply(scan_tensor, spatial_sigma, color_sigma, False)
time_elapsed = time.time()-start
print("completed in: {}".format(time_elapsed))

print('cpu phl')
start=time.time()
result_tensor_cpu_phl = BilateralFilter.apply(scan_tensor, spatial_sigma, color_sigma, True)
time_elapsed = time.time()-start
print("completed in: {}".format(time_elapsed))

print('cuda bruteforce')
start=time.time()
result_tensor_cuda_brute = BilateralFilter.apply(scan_tensor_cuda, spatial_sigma, color_sigma, False)
time_elapsed = time.time()-start
print("completed in: {}".format(time_elapsed))

print('cuda phl')
start=time.time()
result_tensor_cuda_phl = BilateralFilter.apply(scan_tensor_cuda, spatial_sigma, color_sigma, True)
time_elapsed = time.time()-start
print("completed in: {}".format(time_elapsed))

# results
sliceIndex = 64
input_xplane = scan_tensor[0, :, sliceIndex, :, :].permute(1, 2, 0).cpu()
input_yplane = scan_tensor[0, :, :, sliceIndex, :].permute(1, 2, 0).cpu()
input_zplane = scan_tensor[0, :, :, :, sliceIndex].permute(1, 2, 0).cpu()

result_cpu_brute_xplane = result_tensor_cpu_brute[0, :, sliceIndex, :, :].permute(1, 2, 0).cpu()
result_cpu_brute_yplane = result_tensor_cpu_brute[0, :, :, sliceIndex, :].permute(1, 2, 0).cpu()
result_cpu_brute_zplane = result_tensor_cpu_brute[0, :, :, :, sliceIndex].permute(1, 2, 0).cpu()

result_cpu_phl_xplane = result_tensor_cpu_phl[0, :, sliceIndex, :, :].permute(1, 2, 0).cpu()
result_cpu_phl_yplane = result_tensor_cpu_phl[0, :, :, sliceIndex, :].permute(1, 2, 0).cpu()
result_cpu_phl_zplane = result_tensor_cpu_phl[0, :, :, :, sliceIndex].permute(1, 2, 0).cpu()

result_cuda_brute_xplane = result_tensor_cuda_brute[0, :, sliceIndex, :, :].permute(1, 2, 0).cpu()
result_cuda_brute_yplane = result_tensor_cuda_brute[0, :, :, sliceIndex, :].permute(1, 2, 0).cpu()
result_cuda_brute_zplane = result_tensor_cuda_brute[0, :, :, :, sliceIndex].permute(1, 2, 0).cpu()

result_cuda_phl_xplane = result_tensor_cuda_phl[0, :, sliceIndex, :, :].permute(1, 2, 0).cpu()
result_cuda_phl_yplane = result_tensor_cuda_phl[0, :, :, sliceIndex, :].permute(1, 2, 0).cpu()
result_cuda_phl_zplane = result_tensor_cuda_phl[0, :, :, :, sliceIndex].permute(1, 2, 0).cpu()


# plotting
show_result(131, input_xplane)
show_result(132, input_yplane)
show_result(133, input_zplane)
plt.show()

show_result(131, result_cpu_brute_xplane)
show_result(132, result_cpu_brute_yplane)
show_result(133, result_cpu_brute_zplane)
plt.show()

show_result(131, result_cpu_phl_xplane)
show_result(132, result_cpu_phl_yplane)
show_result(133, result_cpu_phl_zplane)
plt.show()

show_result(131, result_cuda_brute_xplane)
show_result(132, result_cuda_brute_yplane)
show_result(133, result_cuda_brute_zplane)
plt.show()

show_result(131, result_cuda_phl_xplane)
show_result(132, result_cuda_phl_yplane)
show_result(133, result_cuda_phl_zplane)
plt.show()
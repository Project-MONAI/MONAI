import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from monai.networks.layers.filtering import BilateralFilter

import nibabel as nib
import matplotlib.pyplot as plt
import time

def run(function, args):
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        result = function(*args)

    cpu_time = sum([item.cpu_time for item in prof.function_events]) / 1e3
    gpu_time = sum([item.cuda_time for item in prof.function_events]) / 1e3
    print("cpu time: {}ms, gpu time: {}ms".format('%.2f' % cpu_time, '%.2f' % gpu_time))
    
    return result

# helper method for plotting
def show_result(position, data):
    plt.subplot(position).axis('off')
    plt.imshow(data, cmap='gray')

# load ct scan data (128x128x128)
scan = nib.load('temp_abdominal_ct.nii').get_fdata()[:, :, 30:-30, 30:-30, 30:-30]
scan += np.min(scan)
scan /= np.max(scan)

# data into tensor
scan_tensor = torch.from_numpy(scan).type(torch.FloatTensor).contiguous()
scan_tensor_cuda = scan_tensor.cuda()

# filter parameters
spatial_sigma = 4
color_sigma = 0.05

# filter
print('running 3D bilateral filter...')

print('cpu bruteforce')
result_tensor_cpu_brute = run(BilateralFilter.apply, (scan_tensor, spatial_sigma, color_sigma, False))

print('cpu phl')
result_tensor_cpu_phl = run(BilateralFilter.apply, (scan_tensor, spatial_sigma, color_sigma, True))

print('cuda bruteforce')
result_tensor_cuda_brute = run(BilateralFilter.apply, (scan_tensor_cuda, spatial_sigma, color_sigma, False))

print('cuda phl')
result_tensor_cuda_phl = run(BilateralFilter.apply, (scan_tensor_cuda, spatial_sigma, color_sigma, True))

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
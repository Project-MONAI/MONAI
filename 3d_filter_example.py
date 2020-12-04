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

def run(function, args):
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        result = function(*args)

    cpu_time = sum([item.cpu_time for item in prof.function_events]) / 1e3
    gpu_time = sum([item.cuda_time for item in prof.function_events]) / 1e3
    print("cpu time: {}ms, gpu time: {}ms".format('%.2f' % cpu_time, '%.2f' % gpu_time))

    sliceIndex = 64
    input_xplane = args[0][0, :, sliceIndex, :, :].permute(1, 2, 0).cpu()
    input_yplane = args[0][0, :, :, sliceIndex, :].permute(1, 2, 0).cpu()
    input_zplane = args[0][0, :, :, :, sliceIndex].permute(1, 2, 0).cpu()

    result_xplane = result[0, :, sliceIndex, :, :].permute(1, 2, 0).cpu()
    result_yplane = result[0, :, :, sliceIndex, :].permute(1, 2, 0).cpu()
    result_zplane = result[0, :, :, :, sliceIndex].permute(1, 2, 0).cpu()
    
    show_result(231, input_xplane)
    show_result(232, input_yplane)
    show_result(233, input_zplane)
    show_result(234, result_xplane)
    show_result(235, result_yplane)
    show_result(236, result_zplane)
    plt.show()

# load ct scan data (128x128x128)
scan = nib.load('temp_abdominal_ct.nii').get_fdata()[:, :, 30:-30, 30:-30, 30:-30]
scan -= np.min(scan)
scan /= np.max(scan)

# data into tensor
scan_tensor = torch.from_numpy(scan).type(torch.FloatTensor).contiguous()
scan_tensor_cuda = scan_tensor.cuda()

# filter parameters
spatial_sigma = 8
color_sigma = 0.2

# filter
print('running 3D bilateral filter...')

print('cpu bruteforce')
run(BilateralFilter.apply, (scan_tensor, spatial_sigma, color_sigma, False))

print('cpu phl')
run(BilateralFilter.apply, (scan_tensor, spatial_sigma, color_sigma, True))

print('cuda bruteforce')
run(BilateralFilter.apply, (scan_tensor_cuda, spatial_sigma, color_sigma, False))

print('cuda phl')
run(BilateralFilter.apply, (scan_tensor_cuda, spatial_sigma, color_sigma, True))

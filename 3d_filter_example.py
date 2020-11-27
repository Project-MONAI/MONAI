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
scan_tensor = torch.from_numpy(scan).type(torch.FloatTensor).contiguous()#.cuda()

# filter parameters
spatial_sigma = 30
color_sigma = 0.05
fast_approx = True

# filter
print('running 3D bilateral filter...')
start=time.time()
result_tensor = BilateralFilter.apply(scan_tensor, spatial_sigma, color_sigma, fast_approx)
time_elapsed = time.time()-start
print("completed in: {}".format(time_elapsed))

# results
sliceIndex = 64
input_xplane = scan_tensor[0, :, sliceIndex, :, :].permute(1, 2, 0).cpu()
input_yplane = scan_tensor[0, :, :, sliceIndex, :].permute(1, 2, 0).cpu()
input_zplane = scan_tensor[0, :, :, :, sliceIndex].permute(1, 2, 0).cpu()

result_xplane = result_tensor[0, :, sliceIndex, :, :].permute(1, 2, 0).cpu()
result_yplane = result_tensor[0, :, :, sliceIndex, :].permute(1, 2, 0).cpu()
result_zplane = result_tensor[0, :, :, :, sliceIndex].permute(1, 2, 0).cpu()

# plotting
show_result(231, input_xplane)
show_result(232, input_yplane)
show_result(233, input_zplane)
show_result(234, result_xplane)
show_result(235, result_yplane)
show_result(236, result_zplane)
plt.show()
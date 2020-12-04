import numpy as np
import torch

from monai.networks.layers.filtering import BilateralFilter

import skimage.data
import matplotlib.pyplot as plt
import time

def run_testcase(test_name, plot_position, function, args):
    print("test case: {}".format(test_name))

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        result = function(*args)

    cpu_time = sum([item.cpu_time for item in prof.function_events]) / 1e3
    gpu_time = sum([item.cuda_time for item in prof.function_events]) / 1e3
    print("cpu time: {}ms, gpu time: {}ms".format('%.2f' % cpu_time, '%.2f' % gpu_time))
    
    result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    plt.subplot(plot_position).axis('off')
    plt.title(test_name)
    plt.imshow(result)

# filter parameters
spatial_sigma = 20
color_sigma = 0.3

# input data
data = np.array(skimage.data.astronaut()) / 255.0
data_tensor = torch.from_numpy(data).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(0).contiguous()
data_tensor_cuda = data_tensor.cuda()

# test cases
print("running tests...")
run_testcase('cpu bruteforce', 221, BilateralFilter.apply, (data_tensor, spatial_sigma, color_sigma, False))
run_testcase('cuda bruteforce', 222, BilateralFilter.apply, (data_tensor_cuda, spatial_sigma, color_sigma, False))
run_testcase('cpu phl', 223, BilateralFilter.apply, (data_tensor, spatial_sigma, color_sigma, True))
run_testcase('cuda phl', 224, BilateralFilter.apply, (data_tensor_cuda, spatial_sigma, color_sigma, True))

plt.show()

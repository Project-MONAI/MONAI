import numpy as np
import torch

from monai.networks.layers.filtering import BilateralFilter

import skimage.data
import matplotlib.pyplot as plt
import time

def run_testcase(test_name, function, args):
    print("test case: {}".format(test_name))

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        result = function(*args)

    cpu_time = sum([item.cpu_time for item in prof.function_events]) / 1e3
    gpu_time = sum([item.cuda_time for item in prof.function_events]) / 1e3
    print("cpu time: {}ms, gpu time: {}ms".format('%.2f' % cpu_time, '%.2f' % gpu_time))
    
    result = result.permute(0, 2, 3, 1).cpu().numpy()
    
    plt.subplot(131).axis('off')
    plt.title(test_name)
    plt.imshow(result[0])
    plt.subplot(132).axis('off')
    plt.title(test_name)
    plt.imshow(result[1])
    plt.subplot(133).axis('off')
    plt.title(test_name)
    plt.imshow(result[2])
    plt.show()

# filter parameters
spatial_sigma = 8
color_sigma = 0.3

# input data
image = np.array(skimage.data.astronaut()) / 255.0
data = np.array([image, image, image]) 
data_tensor = torch.from_numpy(data).type(torch.FloatTensor).permute(0, 3, 1, 2).contiguous()
data_tensor_cuda = data_tensor.cuda()

# test cases
print("running tests...")
# run_testcase('cpu bruteforce', BilateralFilter.apply, (data_tensor, spatial_sigma, color_sigma, False))
# run_testcase('cuda bruteforce', BilateralFilter.apply, (data_tensor_cuda, spatial_sigma, color_sigma, False))
# run_testcase('cpu phl', BilateralFilter.apply, (data_tensor, spatial_sigma, color_sigma, True))
run_testcase('cuda phl', BilateralFilter.apply, (data_tensor_cuda, spatial_sigma, color_sigma, True))


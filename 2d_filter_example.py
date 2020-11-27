import numpy as np
import torch

from monai.networks.layers.filtering import BilateralFilter

import skimage.data
import matplotlib.pyplot as plt
import time

def run_testcase(test_name, plot_position, function, args):
    print("test case: {}".format(test_name))
    start = time.time()
    result = function(*args)
    torch.cuda.synchronize()
    time_elapsed = time.time() - start
    print("completed in: {}".format(time_elapsed))
    
    result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    plt.subplot(plot_position).axis('off')
    plt.title('{}: {}sec'.format(test_name, '%.2f' % time_elapsed))
    plt.imshow(result)

# filter parameters
spatial_sigma = 10
color_sigma = 0.2

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

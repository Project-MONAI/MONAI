import numpy as np
import torch

from monai.networks.layers.filtering import BilateralFilter

import skimage
from skimage.restoration import denoise_bilateral as skimage_bilateral
import matplotlib.pyplot as plt
import time

# filter parameters
spatial_sigma = 10.0
color_sigma = 0.3

# 2d data
data2d = np.array(skimage.data.astronaut()) / 255.0
data2d_tensor = torch.from_numpy(data2d).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(0)

start = time.time()
results2d_tensor = BilateralFilter.apply(data2d_tensor, spatial_sigma, color_sigma)
print(time.time() - start)

results2d = results2d_tensor.squeeze(0).permute(1, 2, 0).numpy()

# results
plt.subplot(121).axis('off')
plt.imshow(data2d)
plt.subplot(122).axis('off')
plt.imshow(results2d)
plt.show()
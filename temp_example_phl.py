import skimage
import matplotlib.pyplot as plt

import numpy as np
import torch 

from monai.networks.layers.permutohedrallattice import PermutohedralLattice as phl

# acquiring test image
image = np.array(skimage.data.astronaut())

# constructing image tensor
image_tensor = torch.from_numpy(image).cuda().permute(2,0,1).unsqueeze(0).unsqueeze(-1)

# constructing feature tensor (x and y coords)
x_coords, y_coords = torch.meshgrid(torch.arange(image.shape[0]).cuda(), torch.arange(image.shape[1]).cuda()) 
feature_tensor = torch.stack([x_coords, y_coords]).unsqueeze(0).unsqueeze(-1).type(torch.cuda.FloatTensor) / 20

# flattening spatial dimensions
input_tensor_shape = image_tensor.size()

image_tensor = torch.flatten(image_tensor, start_dim=2)
feature_tensor = torch.flatten(feature_tensor, start_dim=2)

# applying phl twice, once to acquire a normalisation mask, then again with the real data
normalisation_mask_tensor = phl.apply(feature_tensor, torch.ones_like(image_tensor))
filtered_image_tensor = phl.apply(feature_tensor, image_tensor)

# reshaping output
normalisation_mask_tensor = normalisation_mask_tensor.reshape(input_tensor_shape)
filtered_image_tensor = filtered_image_tensor.reshape(input_tensor_shape)

# reading back data and preparing for display
normalisation_mask = normalisation_mask_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
filtered_image = filtered_image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
filtered_image_normalised = filtered_image / normalisation_mask

# displaying results
plt.subplot(221).axis('off')
plt.title('Input')
plt.imshow(image)
plt.subplot(222).axis('off')
plt.title('Normalisation Mask')
plt.imshow(normalisation_mask / np.max(normalisation_mask))
plt.subplot(223).axis('off')
plt.title('Filtered Image (Un-normalised)')
plt.imshow(filtered_image / np.max(filtered_image))
plt.subplot(224).axis('off')
plt.title('Filtered Image (Normalised)')
plt.imshow(filtered_image_normalised / np.max(filtered_image_normalised))
plt.show()
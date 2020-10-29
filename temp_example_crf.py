import nibabel as nib
import matplotlib.pyplot as plt

import numpy as np
import torch 

from monai.networks.blocks.crf import CRF

# load data
scan = nib.load('.\\temp_abdominal_ct.nii').get_data()
logits = nib.load('.\\temp_abdominal_ct_noisy_logits.nii').get_data()

# taking 10 slices from the middle of the 128x128x128 scan
scan = np.array(scan)[:, :, :, :, 59:-59]
logits = np.array(logits)[:, :, :, :, 59:-59]

# data to GPU
scan_tensor = torch.from_numpy(scan).cuda()
logits_tensor = torch.from_numpy(logits).cuda()

# applying crf 
smoothed_logits_tensor = CRF(num_classes=9).forward(logits_tensor, scan_tensor)

# argmax to convert logits to labels
labels_tensor = torch.argmax(logits_tensor, dim=1)
smoothed_labels_tensor = torch.argmax(smoothed_logits_tensor, dim=1)

# reading back the data
labels = labels_tensor.squeeze().cpu().numpy()
smoothed_labels = smoothed_labels_tensor.squeeze().cpu().numpy()

# displaying results
plt.subplot(121).axis('off')
plt.title('Labels (Noisy Logits)')
plt.imshow(labels[:, :, 5], cmap='Accent')
plt.subplot(122).axis('off')
plt.title('Labels (Smoothed Logits)')
plt.imshow(smoothed_labels[:, :, 5], cmap='Accent')
plt.show()
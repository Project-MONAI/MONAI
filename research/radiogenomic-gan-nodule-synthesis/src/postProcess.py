import os
import sys
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

for i in range(595,596):
        bas = np.asarray(Image.open('Result/BASE_ep%s.png'%i).convert('L'))
        nod = np.asarray(Image.open('Result/Image_ep%s.png'%i).convert('L'))
        seg = np.asarray(Image.open('Result/Seg_ep%s.png'%i).convert('L'))
        bas = bas.astype(float)
        nod = nod.astype(float)
        seg = seg.astype(float)

        seg = gaussian_filter(seg, sigma=5)
        seg = seg/255
        segInv = 1-seg

        out = bas*segInv+nod*seg
        gaussian_filter(out, sigma=1)

        im = Image.fromarray(out.astype(np.uint8))
        im.save("Result/out%s.png"%i)



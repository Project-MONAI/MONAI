# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

from monai.data.streams.datastream import streamgen
from monai.utils.arrayutils import get_valid_patch_size, iter_patch


@streamgen
def select_over_dimension(imgs,dim=-1, indices=None):
    """
    Select and yield data from the images in `imgs` by iterating over the selected dimension. This will yield images
    with one fewer dimension than the inputs.
    
    Args:
        imgs (tuple): tuple of np.ndarrays of 2+ dimensions
        dim (int, optional): dimension to iterate over, default is last dimension
        indices (None or tuple, optional): indices for which arrays in `imgs` to produce patches for, None for all
        
    Yields:
        Arrays chosen the members of `imgs` with one fewer dimension, iterating over dimension `dim` in order
    """
    # select only certain images to iterate over
    indices = indices or list(range(len(imgs)))
    imgs = [imgs[i] for i in indices]
    
    slices=[slice(None)]*imgs[0].ndim # define slices selecting the whole image
    
    for i in range(imgs[0].shape[dim]):
        slices[dim]=i # select index in dimension 
        yield tuple(im[tuple(slices)] for im in imgs)
        

@streamgen
def uniform_random_patches(imgs, patch_size=64, num_patches=10, indices=None):
    """
    Choose patches from the input image(s) of a given size at random. The choice of patch position is uniformly
    distributed over the image.
    
    Args:
        imgs (tuple): tuple of np.ndarrays of 2+ dimensions
        patch_size (int or tuple, optional): a single dimension or a tuple of dimension indicating the patch size, this
            can be a different dimensionality from the source image to produce smaller dimension patches, and None or 0
            can be used to select the whole dimension from the input image
        num_patches (int, optional): number of patches to produce per image set
        indices (None or tuple, optional): indices for which arrays in `imgs` to produce patches for, None for all
        
    Yields:
        Patches from the source image(s) from uniformly random positions of size specified by `patch_size`
    """

    # select only certain images to iterate over
    indices = indices or list(range(len(imgs)))
    imgs = [imgs[i] for i in indices]

    patch_size = get_valid_patch_size(imgs[0].shape, patch_size)

    for _ in range(num_patches):
        # choose the minimal corner of the patch to yield
        min_corner = tuple(np.random.randint(0, ms - ps) if ms > ps else 0 for ms, ps in zip(imgs[0].shape, patch_size))

        # create the slices for each dimension which define the patch in the source volume
        slices = tuple(slice(mc, mc + ps) for mc, ps in zip(min_corner, patch_size))

        # select out a patch from each image volume
        yield tuple(im[slices] for im in imgs)


@streamgen
def ordered_patches(imgs, patch_size=64, start_pos=(), indices=None, pad_mode="wrap", **pad_opts):
    """
    Choose patches from the input image(s) of a given size in a contiguous grid. Patches are selected iterating by the 
    patch size in the first dimension, followed by second, etc. This allows the sampling of images in a uniform grid-
    wise manner that ensures the whole image is visited. The images can be padded to include margins if the patch size
    is not an even multiple of the image size. A start position can also be specified to start the iteration from a
    position other than 0.
    
    Args:
        imgs (tuple): tuple of np.ndarrays of 2+ dimensions
        patch_size (int or tuple, optional): a single dimension or a tuple of dimension indicating the patch size, this
            can be a different dimensionality from the source image to produce smaller dimension patches, and None or 0
            can be used to select the whole dimension from the input image
        start_pos (tuple, optional): starting position in the image, default is 0 in each dimension
        indices (None or tuple, optional): indices for which arrays in `imgs` to produce patches for, None for all
        pad_mode (str, optional): padding mode, see numpy.pad
        pad_opts (dict, optional): padding options, see numpy.pad
        
    Yields:
        Patches from the source image(s) in grid ordering of size specified by `patch_size`
    """
    
    # select only certain images to iterate over
    indices = indices or list(range(len(imgs)))
    imgs = [imgs[i] for i in indices]

    iters = [iter_patch(i, patch_size, start_pos, False, pad_mode, **pad_opts) for i in imgs]

    yield from zip(*iters)

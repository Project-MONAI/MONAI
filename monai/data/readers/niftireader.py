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
import nibabel as nib

from monai.data.streams.datastream import LRUCacheStream
from monai.utils.moduleutils import export


def load_nifti(filename_or_obj, as_closest_canonical=False, image_only=True, dtype=None):
    """
    Loads a Nifti file from the given path or file-like object.

    Args:
        filename_or_obj (str or file): path to file or file-like object
        as_closest_canonical (bool): if True, load the image as closest to canonical axis format
        image_only (bool): if True return only the image volume, other return image volume and header dict
        dtype (np.dtype, optional): if not None convert the loaded image to this data type

    Returns:
        The loaded image volume if `image_only` is True, or a tuple containing the volume and the Nifti
        header in dict format otherwise
    """
    img = nib.load(filename_or_obj)

    if as_closest_canonical:
        img = nib.as_closest_canonical(img)

    if dtype is not None:
        dat = img.get_fdata(dtype=dtype)
    else:
        dat = np.asanyarray(img.dataobj)

    header = dict(img.header)
    header['filename_or_obj'] = filename_or_obj

    if image_only:
        return dat
    else:
        return dat, header


@export("monai.data.readers")
class NiftiCacheReader(LRUCacheStream):
    """
    Read Nifti files from incoming file names. Multiple filenames for data item can be defined which will load
    multiple Nifti files. As this inherits from CacheStream this will cache nifti image volumes in their entirety.
    The arguments for load() other than `names` must be passed to the constructor.

    Args:
        src (Iterable): source iterable object
        indices (tuple or None, optional): indices of values from source to load
        as_closest_canonical (bool): if True, load the image as closest to canonical axis format
        image_only (bool): if True return only the image volume, other return image volume and header dict
        dtype (np.dtype, optional): if not None convert the loaded image to this data type
    """

    def load(self, names, indices=None, as_closest_canonical=False, image_only=True, dtype=None):
        if isinstance(names, str):
            names = [names]
            indices = [0]
        else:
            # names may be a tuple containing a single np.ndarray containing file names
            if len(names) == 1 and not isinstance(names[0], str):
                names = names[0]

            indices = indices or list(range(len(names)))

        filenames = [names[i] for i in indices]
        result = tuple(load_nifti(f, as_closest_canonical, image_only, dtype) for f in filenames)

        return result if len(result) > 1 else result[0]

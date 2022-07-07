# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray

from monai.config import PathLike
from monai.data.image_reader import ImageReader
from monai.data.utils import is_supported_format
from monai.utils import optional_import, require_pkg
from monai.utils.type_conversion import convert_to_tensor

if TYPE_CHECKING:
    import itk
    import nibabel as nib
    import nrrd
    import pydicom
    from nibabel.nifti1 import Nifti1Image
    from PIL import Image as PILImage

    has_nrrd = has_itk = has_nib = has_pil = has_pydicom = True
else:
    itk, has_itk = optional_import("itk", allow_namespace_pkg=True)
    nib, has_nib = optional_import("nibabel")
    Nifti1Image, _ = optional_import("nibabel.nifti1", name="Nifti1Image")
    PILImage, has_pil = optional_import("PIL.Image")
    pydicom, has_pydicom = optional_import("pydicom")
    nrrd, has_nrrd = optional_import("nrrd", allow_namespace_pkg=True)

OpenSlide, _ = optional_import("openslide", name="OpenSlide")
CuImage, _ = optional_import("cucim", name="CuImage")
TiffFile, _ = optional_import("tifffile", name="TiffFile")


if TYPE_CHECKING:
    import h5py

    has_h5py = True
else:
    h5py, has_h5py = optional_import("h5py")


@require_pkg(pkg_name="h5py")
class FastMRIReader(ImageReader):
    """
    Load fastMRI files with '.h5' suffix.
    """

    def __init__(self):
        super().__init__()

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
         Verify whether the specified file format is supported by h5py reader.

        Args:
             filename: file name
        """
        suffixes: Sequence[str] = [".h5"]
        return has_h5py and is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike]) -> Dict:  # type: ignore
        """
        Read data from specified h5 file.
        Note that the returned object is a dictionary.

        Args:
            data: file name to read.
        """
        if isinstance(data, (tuple, list)):
            data = data[0]

        with h5py.File(data, "r") as f:
            # extract everything from the ht5 file
            dat = dict(
                [(key, f[key][()]) for key in f]
                + [(key, f.attrs[key]) for key in f.attrs]
                + [("filename", str(data).split("/")[-1])]
            )
        f.close()

        return dat

    def get_data(self, dat: Dict) -> Tuple[ndarray, dict]:
        """
        Extract data array and metadata from the loaded data and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.

        Args:
            dat: a dictionary loaded from an h5 file
        """
        header = self._get_meta_dict(dat)
        data: ndarray = np.array(dat["kspace"])
        header["mask"] = (
            convert_to_tensor(np.array(dat["mask"])).unsqueeze(0)[None, ..., None]
            if "mask" in dat.keys()
            else np.zeros(data.shape)
        )
        return data, header

    def _get_meta_dict(self, dat) -> Dict:
        """
        Get all the metadata of the loaded dict and return the meta dict.

        Args:
            dat: a dictionary object loaded from an h5 file.
        """
        return {
            "filename": dat["filename"],
            "reconstruction_rss": dat["reconstruction_rss"],
            "acquisition": dat["acquisition"],
            "max": dat["max"],
            "norm": dat["norm"],
            "patient_id": dat["patient_id"],
        }

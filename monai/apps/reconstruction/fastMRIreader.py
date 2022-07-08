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

import subprocess
import sys
from typing import Dict, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray

from monai.config import PathLike
from monai.data.image_reader import ImageReader
from monai.data.utils import is_supported_format
from monai.utils import require_pkg  # optional_import
from monai.utils import FastMRIKeys
from monai.utils.type_conversion import convert_to_tensor

# h5py should be added to monai requirements so that we can use the following 5 lines
# if TYPE_CHECKING:
#    import h5py

#    has_h5py = True
# else:
# h5py, has_h5py = optional_import("h5py")  # ideally should use this

has_h5py = False
if has_h5py:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py"])
    import h5py

    has_h5py = True


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
        data: ndarray = np.array(dat[FastMRIKeys.KSPACE])
        header["mask"] = (
            convert_to_tensor(np.array(dat[FastMRIKeys.MASK])).unsqueeze(0)[None, ..., None]
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
            "filename": dat[FastMRIKeys.FILENAME],
            "reconstruction_rss": dat[FastMRIKeys.RECON],
            "acquisition": dat[FastMRIKeys.ACQUISITION],
            "max": dat[FastMRIKeys.MAX],
            "norm": dat[FastMRIKeys.NORM],
            "patient_id": dat[FastMRIKeys.PID],
        }

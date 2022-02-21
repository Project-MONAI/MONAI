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

import os
import tempfile
import unittest

from monai.transforms import Compose, CopyItemsd, EnsureChannelFirstd, Lambda, LoadImaged, ResampleToMatchd, SaveImaged
from tests.utils import assert_allclose, download_url_or_skip_test


def update_fname(d):
    d["im3_meta_dict"]["filename_or_obj"] = "file3.nii.gz"
    return d


class TestResampleToMatchd(unittest.TestCase):
    def test_correct(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            url_1 = (
                "https://github.com/rcuocolo/PROSTATEx_masks/raw/master/Files/"
                + "lesions/Images/T2/ProstateX-0000_t2_tse_tra_4.nii.gz"
            )
            url_2 = (
                "https://github.com/rcuocolo/PROSTATEx_masks/raw/master/Files/"
                + "lesions/Images/ADC/ProstateX-0000_ep2d_diff_tra_7.nii.gz"
            )
            fname_1 = os.path.join(temp_dir, "file1.nii.gz")
            fname_2 = os.path.join(temp_dir, "file2.nii.gz")
            md5_1 = "adb3f1c4db66a6481c3e4a2a3033c7d5"
            md5_2 = "f12a11ad0ebb0b1876e9e010564745d2"
            download_url_or_skip_test(url=url_1, filepath=fname_1, hash_val=md5_1)
            download_url_or_skip_test(url=url_2, filepath=fname_2, hash_val=md5_2)

            transforms = Compose(
                [
                    LoadImaged(("im1", "im2")),
                    EnsureChannelFirstd(("im1", "im2")),
                    CopyItemsd(("im2", "im2_meta_dict"), names=("im3", "im3_meta_dict")),
                    ResampleToMatchd("im3", "im1_meta_dict"),
                    Lambda(update_fname),
                    SaveImaged("im3", output_dir=temp_dir, output_postfix="", separate_folder=False),
                ]
            )
            data = transforms({"im1": fname_1, "im2": fname_2})
            assert_allclose(data["im1"].shape, data["im3"].shape)


if __name__ == "__main__":
    unittest.main()

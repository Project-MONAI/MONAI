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

from monai.transforms import (
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    Invertd,
    Lambda,
    LoadImaged,
    ResampleToMatchd,
    SaveImaged,
)
from tests.utils import assert_allclose, download_url_or_skip_test, testing_data_config


def update_fname(d):
    d["im3_meta_dict"]["filename_or_obj"] = "file3.nii.gz"
    return d


class TestResampleToMatchd(unittest.TestCase):
    def setUp(self):
        self.fnames = []
        for key in ("0000_t2_tse_tra_4", "0000_ep2d_diff_tra_7"):
            fname = os.path.join(os.path.dirname(__file__), "testing_data", f"test_{key}.nii.gz")
            url = testing_data_config("images", key, "url")
            hash_type = testing_data_config("images", key, "hash_type")
            hash_val = testing_data_config("images", key, "hash_val")
            download_url_or_skip_test(url=url, filepath=fname, hash_type=hash_type, hash_val=hash_val)
            self.fnames.append(fname)

    def test_correct(self):
        with tempfile.TemporaryDirectory() as temp_dir:
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
            data = transforms({"im1": self.fnames[0], "im2": self.fnames[1]})
            # check that output sizes match
            assert_allclose(data["im1"].shape, data["im3"].shape)
            # and that the meta data has been updated accordingly
            assert_allclose(data["im3"].shape[1:], data["im3_meta_dict"]["spatial_shape"], type_test=False)
            assert_allclose(data["im3_meta_dict"]["affine"], data["im1_meta_dict"]["affine"])
            # check we're different from the original
            self.assertTrue(any(i != j for i, j in zip(data["im3"].shape, data["im2"].shape)))
            self.assertTrue(
                any(
                    i != j
                    for i, j in zip(
                        data["im3_meta_dict"]["affine"].flatten(), data["im2_meta_dict"]["affine"].flatten()
                    )
                )
            )
            # test the inverse
            data = Invertd("im3", transforms, "im3")(data)
            assert_allclose(data["im2"].shape, data["im3"].shape)


if __name__ == "__main__":
    unittest.main()

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

import copy
import itertools
import os
import tempfile
import unittest

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.data.image_reader import ITKReader, NibabelReader
from monai.data.image_writer import ITKWriter
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ResampleToMatch, SaveImaged
from tests.utils import assert_allclose, download_url_or_skip_test, testing_data_config

TEST_CASES = ["itkreader", "nibabelreader"]


class TestResampleToMatch(unittest.TestCase):
    def setUp(self):
        self.fnames = []
        for key in ("0000_t2_tse_tra_4", "0000_ep2d_diff_tra_7"):
            fname = os.path.join(os.path.dirname(__file__), "testing_data", f"test_{key}.nii.gz")
            url = testing_data_config("images", key, "url")
            hash_type = testing_data_config("images", key, "hash_type")
            hash_val = testing_data_config("images", key, "hash_val")
            download_url_or_skip_test(url=url, filepath=fname, hash_type=hash_type, hash_val=hash_val)
            self.fnames.append(fname)

    @parameterized.expand(itertools.product([NibabelReader, ITKReader], ["monai.data.NibabelWriter", ITKWriter]))
    def test_correct(self, reader, writer):
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = Compose([LoadImaged(("im1", "im2"), reader=reader), EnsureChannelFirstd(("im1", "im2"))])
            data = loader({"im1": self.fnames[0], "im2": self.fnames[1]})

            with self.assertRaises(ValueError):
                ResampleToMatch(mode=None)(data["im2"], data["im2_meta_dict"], data["im1_meta_dict"])
            im_mod, meta = ResampleToMatch()(data["im2"], data["im2_meta_dict"], data["im1_meta_dict"])
            current_dims = copy.deepcopy(meta.get("dim"))
            saver = SaveImaged("im3", output_dir=temp_dir, output_postfix="", separate_folder=False, writer=writer)
            meta["filename_or_obj"] = "file3.nii.gz"
            saver({"im3": im_mod, "im3_meta_dict": meta})

            saved = nib.load(os.path.join(temp_dir, meta["filename_or_obj"]))
            assert_allclose(data["im1"].shape[1:], saved.shape)
            assert_allclose(saved.header["dim"][:4], np.array([3, 384, 384, 19]))
            if current_dims is not None:
                assert_allclose(saved.header["dim"], current_dims)


if __name__ == "__main__":
    unittest.main()

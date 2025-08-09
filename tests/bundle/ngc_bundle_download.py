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

from __future__ import annotations

import os
import tempfile
import unittest

import torch
from parameterized import parameterized

from monai.apps import check_hash
from monai.bundle import download, load
from tests.test_utils import assert_allclose, skip_if_downloading_fails, skip_if_quick, skip_if_windows

TEST_CASE_NGC_1 = [
    "spleen_ct_segmentation",
    "0.3.7",
    None,
    "monai_spleen_ct_segmentation",
    "models/model.pt",
    "b418a2dc8672ce2fd98dc255036e7a3d",
]
TEST_CASE_NGC_2 = [
    "monai_spleen_ct_segmentation",
    "0.3.7",
    "monai_",
    "spleen_ct_segmentation",
    "models/model.pt",
    "b418a2dc8672ce2fd98dc255036e7a3d",
]

TESTCASE_WEIGHTS = {
    "key": "model.0.conv.unit0.adn.N.bias",
    "value": torch.tensor(
        [
            -0.0705,
            -0.0937,
            -0.0422,
            -0.2068,
            0.1023,
            -0.2007,
            -0.0883,
            0.0018,
            -0.1719,
            0.0116,
            0.0285,
            -0.0044,
            0.1223,
            -0.1287,
            -0.1858,
            0.0460,
        ]
    ),
}


@skip_if_windows
class TestNgcBundleDownload(unittest.TestCase):
    @parameterized.expand([TEST_CASE_NGC_1, TEST_CASE_NGC_2])
    @skip_if_quick
    def test_ngc_download_bundle(self, bundle_name, version, remove_prefix, download_name, file_path, hash_val):
        with skip_if_downloading_fails():
            with tempfile.TemporaryDirectory() as tempdir:
                download(
                    name=bundle_name, source="ngc", version=version, bundle_dir=tempdir, remove_prefix=remove_prefix
                )
                full_file_path = os.path.join(tempdir, download_name, file_path)
                self.assertTrue(os.path.exists(full_file_path))
                self.assertTrue(check_hash(filepath=full_file_path, val=hash_val))

                model = load(
                    name=bundle_name, source="ngc", version=version, bundle_dir=tempdir, remove_prefix=remove_prefix
                )
                assert_allclose(
                    model.state_dict()[TESTCASE_WEIGHTS["key"]],
                    TESTCASE_WEIGHTS["value"],
                    atol=1e-4,
                    rtol=1e-4,
                    type_test=False,
                )


if __name__ == "__main__":
    unittest.main()

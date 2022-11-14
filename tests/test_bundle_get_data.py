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

import unittest

from parameterized import parameterized

from monai.bundle import get_all_bundles_list, get_bundle_info, get_bundle_versions
from monai.utils import optional_import
from tests.utils import SkipIfNoModule, skip_if_downloading_fails, skip_if_quick, skip_if_windows

requests, _ = optional_import("requests")

TEST_CASE_1 = [{"bundle_name": "brats_mri_segmentation"}]

TEST_CASE_2 = [{"bundle_name": "spleen_ct_segmentation", "version": "0.1.0", "auth_token": None}]

TEST_CASE_FAKE_TOKEN = [{"bundle_name": "spleen_ct_segmentation", "version": "0.1.0", "auth_token": "ghp_errortoken"}]


@skip_if_windows
@SkipIfNoModule("requests")
class TestGetBundleData(unittest.TestCase):
    @skip_if_quick
    def test_get_all_bundles_list(self):
        with skip_if_downloading_fails():
            output = get_all_bundles_list()
            self.assertTrue(isinstance(output, list))
            self.assertTrue(isinstance(output[0], tuple))
            self.assertTrue(len(output[0]) == 2)

    @parameterized.expand([TEST_CASE_1])
    @skip_if_quick
    def test_get_bundle_versions(self, params):
        with skip_if_downloading_fails():
            output = get_bundle_versions(**params)
            self.assertTrue(isinstance(output, dict))
            self.assertTrue("latest_version" in output and "all_versions" in output)
            self.assertTrue("0.1.0" in output["all_versions"])

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    @skip_if_quick
    def test_get_bundle_info(self, params):
        with skip_if_downloading_fails():
            output = get_bundle_info(**params)
            self.assertTrue(isinstance(output, dict))
            for key in ["id", "name", "size", "download_count", "browser_download_url"]:
                self.assertTrue(key in output)

    @parameterized.expand([TEST_CASE_FAKE_TOKEN])
    @skip_if_quick
    def test_fake_token(self, params):
        with skip_if_downloading_fails():
            with self.assertRaises(requests.exceptions.HTTPError):
                get_bundle_info(**params)


if __name__ == "__main__":
    unittest.main()

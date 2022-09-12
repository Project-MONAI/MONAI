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
from tests.utils import skip_if_downloading_fails, skip_if_quick, skip_if_windows

TEST_CASE_1 = [{"bundle_name": "brats_mri_segmentation"}]

TEST_CASE_2 = [{"bundle_name": "spleen_ct_segmentation", "version": "0.1.0"}]


@skip_if_windows
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


if __name__ == "__main__":
    unittest.main()

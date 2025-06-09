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

import unittest

from monai.apps.reconstruction.transforms.dictionary import ExtractDataKeyFromMetaKeyd
from monai.data import MetaTensor


class TestExtractDataKeyFromMetaKeyd(unittest.TestCase):
    def test_extract_data_key_from_dic(self):
        data = {"image_data": MetaTensor([1, 2, 3]), "foo_meta_dict": {"filename_or_obj": "test_image.nii.gz"}}

        extract = ExtractDataKeyFromMetaKeyd("filename_or_obj", meta_key="foo_meta_dict")
        result = extract(data)

        assert data["foo_meta_dict"]["filename_or_obj"] == result["filename_or_obj"]

    def test_extract_data_key_from_meta_tensor(self):
        data = {"image_data": MetaTensor([1, 2, 3], meta={"filename_or_obj": 1})}

        extract = ExtractDataKeyFromMetaKeyd("filename_or_obj", meta_key="image_data")
        result = extract(data)

        assert data["image_data"].meta["filename_or_obj"] == result["filename_or_obj"]


if __name__ == "__main__":
    unittest.main()

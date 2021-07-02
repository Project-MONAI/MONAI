# Copyright 2020 - 2021 MONAI Consortium
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
from urllib.error import ContentTooShortError, HTTPError

import numpy as np
import torch
from parameterized import parameterized

from monai.apps import download_mmar, load_from_mmar
from monai.apps.mmars import MODEL_DESC
from monai.apps.mmars.mmars import _get_val
from tests.utils import SkipIfAtLeastPyTorchVersion, SkipIfBeforePyTorchVersion, skip_if_quick

TEST_CASES = [["clara_pt_prostate_mri_segmentation_1"], ["clara_pt_covid19_ct_lesion_segmentation_1"]]
TEST_EXTRACT_CASES = [
    (
        {
            "item": "clara_pt_prostate_mri_segmentation_1",
            "map_location": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "UNet",
        np.array(
            [
                [[-0.0838, 0.0116, -0.0861], [-0.0792, 0.2216, -0.0301], [-0.0379, 0.0006, -0.0399]],
                [[-0.0347, 0.0979, 0.0754], [0.1689, 0.3759, 0.2584], [-0.0698, 0.2740, 0.1414]],
                [[-0.0772, 0.1046, -0.0103], [0.0917, 0.1942, 0.0284], [-0.0165, -0.0181, 0.0247]],
            ]
        ),
    ),
    (
        {
            "item": "clara_pt_covid19_ct_lesion_segmentation_1",
            "map_location": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "SegResNet",
        np.array(
            [
                [
                    [-0.21147135, 0.10815059, -0.04733997],
                    [-0.3425553, 0.03304602, 0.113512],
                    [0.1278807, 0.26298857, -0.0583012],
                ],
                [
                    [-0.3658006, -0.14725913, 0.01149207],
                    [-0.5453718, -0.12894264, -0.05492746],
                    [0.16887102, 0.17586298, 0.03977356],
                ],
                [
                    [-0.12767333, -0.07876065, 0.03136465],
                    [0.26057404, -0.03538669, 0.07552322],
                    [0.23879515, 0.04919613, 0.01725162],
                ],
            ]
        ),
    ),
    (
        {
            "item": "clara_pt_fed_learning_brain_tumor_mri_segmentation_1",
            "map_location": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "SegResNet",
        np.array(
            [
                [[-0.0839, 0.0715, -0.0760], [0.0645, 0.1186, 0.0218], [0.0303, 0.0631, -0.0648]],
                [[0.0128, 0.1440, 0.0213], [0.1658, 0.1813, 0.0541], [-0.0627, 0.0839, 0.0660]],
                [[-0.1207, 0.0138, -0.0808], [0.0277, 0.0416, 0.0597], [0.0455, -0.0134, -0.0949]],
            ]
        ),
    ),
    (
        {
            "item": "clara_pt_pathology_metastasis_detection_1",
            "map_location": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "TorchVisionFullyConvModel",
        np.array(
            [
                [-0.00693138, -0.00441378, -0.01057985, 0.05604396, 0.03526996, -0.00399302, -0.0267504],
                [0.00805358, 0.01016939, -0.10749951, -0.28787708, -0.27905375, -0.13328083, -0.00882593],
                [-0.01909848, 0.04871106, 0.2957697, 0.60376877, 0.53552634, 0.24821444, 0.03773781],
                [0.02449462, -0.07471243, -0.30943492, -0.43987238, -0.26549947, -0.00698426, 0.04395606],
                [-0.03124012, 0.00807883, 0.06797771, -0.04612541, -0.30266526, -0.39722857, -0.25109962],
                [0.02480375, 0.03378576, 0.06519791, 0.24546203, 0.41867673, 0.393786, 0.16055048],
                [-0.01529332, -0.00062494, -0.016658, -0.06313603, -0.1508078, -0.09107386, -0.01239121],
            ]
        ),
    ),
]


class TestMMMARDownload(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    @skip_if_quick
    @SkipIfBeforePyTorchVersion((1, 6))
    def test_download(self, idx):
        try:
            download_mmar(idx)
            download_mmar(idx, progress=False)  # repeated to check caching
            with tempfile.TemporaryDirectory() as tmp_dir:
                download_mmar(idx, mmar_dir=tmp_dir, progress=False)
                download_mmar(idx, mmar_dir=tmp_dir, progress=False, version=1)  # repeated to check caching
                self.assertTrue(os.path.exists(os.path.join(tmp_dir, idx)))
        except (ContentTooShortError, HTTPError, RuntimeError) as e:
            print(str(e))
            if isinstance(e, HTTPError):
                self.assertTrue("500" in str(e))  # http error has the code 500
            return  # skipping this test due the network connection errors

    @parameterized.expand(TEST_EXTRACT_CASES)
    @skip_if_quick
    @SkipIfBeforePyTorchVersion((1, 6))
    def test_load_ckpt(self, input_args, expected_name, expected_val):
        try:
            output = load_from_mmar(**input_args)
        except (ContentTooShortError, HTTPError, RuntimeError) as e:
            print(str(e))
            if isinstance(e, HTTPError):
                self.assertTrue("500" in str(e))  # http error has the code 500
            return
        self.assertEqual(output.__class__.__name__, expected_name)
        x = next(output.parameters())  # verify the first element
        np.testing.assert_allclose(x[0][0].detach().cpu().numpy(), expected_val, rtol=1e-3, atol=1e-3)

    def test_unique(self):
        # model ids are unique
        keys = sorted([m["id"] for m in MODEL_DESC])
        self.assertTrue(keys == sorted(set(keys)))

    @SkipIfAtLeastPyTorchVersion((1, 6))
    def test_no_default(self):
        with self.assertRaises(ValueError):
            download_mmar(0)

    def test_search(self):
        self.assertEqual(_get_val({"a": 1, "b": 2}, key="b"), 2)
        self.assertEqual(_get_val({"a": {"c": {"c": 4}}, "b": {"c": 2}}, key="b"), {"c": 2})
        self.assertEqual(_get_val({"a": {"c": 4}, "b": {"c": 2}}, key="c"), 4)
        self.assertEqual(_get_val({"a": {"c": None}, "b": {"c": 2}}, key="c"), 2)


if __name__ == "__main__":
    unittest.main()

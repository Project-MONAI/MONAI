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

import json
import os
import tempfile
import unittest

from parameterized import parameterized

TEST_CASE_1 = [
    {
        "version": "0.1.0",
        "changelog": {
            "0.1.0": "complete the model package",
            "0.0.1": "initialize the model package structure"
        },
        "monai_version": "0.8.0",
        "pytorch_version": "1.10.0",
        "numpy_version": "1.21.2",
        "optional_packages_version": {
            "nibabel": "3.2.1"
        },
        "task": "Decathlon spleen segmentation",
        "description": "A pre-trained model for volumetric (3D) segmentation of the spleen from CT image",
        "authorship": "MONAI team",
        "copyright": "Copyright (c) MONAI Consortium",
        "data_source": "Task09_Spleen.tar from http://medicaldecathlon.com/",
        "data_type": "dicom",
        "dataset_dir": "/workspace/data/Task09_Spleen",
        "image_classes": "single channel data, intensity scaled to [0, 1]",
        "label_classes": "single channel data, 1 is spleen, 0 is everything else",
        "pred_classes": "2 channels OneHot data, channel 1 is spleen, channel 0 is background",
        "eval_metrics": {
            "mean_dice": 0.96
        },
        "intended_use": "This is an example, not to be used for diagnostic purposes",
        "references": [
            "Xia, Yingda, et al. '3D Semi-Supervised Learning with Uncertainty-Aware Multi-View Co-Training.'"
            " arXiv preprint arXiv:1811.12506 (2018). https://arxiv.org/abs/1811.12506.",
            "Kerfoot E., Clough J., Oksuz I., Lee J., King A.P., Schnabel J.A. (2019)"
            " Left-Ventricle Quantification Using Residual U-Net. In: Pop M. et al. (eds) Statistical Atlases"
            " and Computational Models of the Heart. Atrial Segmentation and LV Quantification Challenges. STACOM 2018."
            " Lecture Notes in Computer Science, vol 11395. Springer, Cham. https://doi.org/10.1007/978-3-030-12029-0_40"
        ],
        "network_data_format": {
            "inputs": {
                "image": {
                    "type": "image",
                    "format": "magnitude",
                    "num_channels": 1,
                    "spatial_shape": [
                        160,
                        160,
                        160
                    ],
                    "dtype": "float32",
                    "value_range": [
                        0,
                        1
                    ],
                    "is_patch_data": False,
                    "channel_def": {
                        "0": "image"
                    }
                }
            },
            "outputs": {
                "pred": {
                    "type": "image",
                    "format": "segmentation",
                    "num_channels": 2,
                    "spatial_shape": [
                        160,
                        160,
                        160
                    ],
                    "dtype": "float32",
                    "value_range": [
                        0,
                        1
                    ],
                    "is_patch_data": False,
                    "channel_def": {
                        "0": "background",
                        "1": "spleen"
                    }
                }
            }
        }
    }
]


class TestVerifyMeta(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_verify(self, meta_data):
        with tempfile.TemporaryDirectory() as tempdir:
            #filepath = os.path.join(tempdir, "schema.json")
            filepath = "/workspace/data/medical/MONAI/monai/apps/manifest/metadata.json"

            metafile = os.path.join(tempdir, "metadata.json")
            with open(metafile, "w") as f:
                json.dump(meta_data, f)

            os.system(f"python -m monai.apps.manifest.verify_meta -m {metafile} -u fsdfsfs -f {filepath}")


if __name__ == "__main__":
    unittest.main()

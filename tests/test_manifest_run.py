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
import logging
import sys
import os
import unittest
import numpy as np
import tempfile
import nibabel as nib

from parameterized import parameterized
from monai.transforms import LoadImage

TEST_CASE_1 = [
    {
        "device": "$torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "network_def": {
            "<name>": "UNet",
            "<args>": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 2,
                "channels": [16, 32, 64, 128, 256],
                "strides": [2, 2, 2, 2],
                "num_res_units": 2,
                "norm": "batch"
            }
        },
        "network": "$@network_def.to(@device)",
        "preprocessing": {
            "<name>": "Compose",
            "<args>": {
                "transforms": [
                    {
                        "<name>": "LoadImaged",
                        "<args>": {
                            "keys": "image"
                        }
                    },
                    {
                        "<name>": "EnsureChannelFirstd",
                        "<args>": {
                            "keys": "image"
                        }
                    },
                    {
                        "<name>": "ScaleIntensityd",
                        "<args>": {
                            "keys": "image"
                        }
                    },
                    {
                        "<name>": "EnsureTyped",
                        "<args>": {
                            "keys": "image"
                        }
                    }
                ]
            }
        },
        "dataset": {
            "<name>": "Dataset",
            "<args>": {
                "data": "@<meta>#datalist",  # test placeholger with `datalist`
                "transform": "@preprocessing"
            }
        },
        "dataloader": {
            "<name>": "DataLoader",
            "<args>": {
                "dataset": "@dataset",
                "batch_size": 1,
                "shuffle": False,
                "num_workers": 4
            }
        },
        "inferer": {
            "<name>": "SlidingWindowInferer",
            "<args>": {
                "roi_size": [96, 96, 96],
                "sw_batch_size": 4,
                "overlap": 0.5
            }
        },
        "postprocessing": {
            "<name>": "Compose",
            "<args>": {
                "transforms": [
                    {
                        "<name>": "Activationsd",
                        "<args>": {
                            "keys": "pred",
                            "softmax": True
                        }
                    },
                    {
                        "<name>": "AsDiscreted",
                        "<args>": {
                            "keys": "pred",
                            "argmax": True
                        }
                    },
                    {
                        "<name>": "SaveImaged",
                        "<args>": {
                            "keys": "pred",
                            "meta_keys": "image_meta_dict",
                            "output_dir": "@<meta>#output_dir"  # test placeholger with `output_dir`
                        }
                    }
                ]
            }
        },
        "evaluator": {
            "<name>": "SupervisedEvaluator",
            "<args>": {
                "device": "@device",
                "val_data_loader": "@dataloader",
                "network": "@network",
                "inferer": "@inferer",
                "postprocessing": "@postprocessing",
                "amp": False
            }
        }
    },
    (128, 128, 128),
]


class TestChannelPad(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_shape(self, config, expected_shape):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        test_image = np.random.rand(128, 128, 128)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "image.nii")
            nib.save(nib.Nifti1Image(test_image, np.eye(4)), filename)

            meta = {
                "datalist": [{"image": filename}],
                "output_dir": tempdir,
                "window": (96, 96, 96),
            }
            metafile = os.path.join(tempdir, "meta.json")
            with open(metafile, "w") as f:
                json.dump(meta, f)

            configfile = os.path.join(tempdir, "config.json")
            with open(configfile, "w") as f:
                json.dump(config, f)

            os.system(
                f"python -m monai.apps.manifest.run -m {metafile} -c {configfile}"
                f" -o 'evaluator#<args>#amp'=False -t evaluator"
            )

            saved = LoadImage(image_only=True)(os.path.join(tempdir, "image", "image_trans.nii.gz"))
            self.assertTupleEqual(saved.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()

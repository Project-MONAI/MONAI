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
from copy import deepcopy

from monai.apps.auto3dseg import BundleAlgo
from monai.bundle import ConfigParser


class Segresnet2DAlgo(BundleAlgo):
    def fill_template_config(self, data_stats):
        if data_stats is None:
            return
        data_cfg = ConfigParser(globals=False)
        if os.path.exists(str(data_stats)):
            data_cfg.read_config(str(data_stats))
        else:
            data_cfg.update(data_stats)
        patch_size = [320, 320]
        max_shape = data_cfg["stats_summary#image_stats#shape#max"]
        patch_size = [
            max(32, shape_k // 32 * 32) if shape_k < p_k else p_k for p_k, shape_k in zip(patch_size, max_shape)
        ]
        self.cfg["patch_size#0"], self.cfg["patch_size#1"] = patch_size
        self.cfg["patch_size_valid#0"], self.cfg["patch_size_valid#1"] = patch_size
        data_src_cfg = ConfigParser(globals=False)
        if self.data_list_file is not None and os.path.exists(str(self.data_list_file)):
            data_src_cfg.read_config(self.data_list_file)
            self.cfg.update(
                {
                    "data_file_base_dir": data_src_cfg["dataroot"],
                    "data_list_file_path": data_src_cfg["datalist"],
                    "input_channels": data_cfg["stats_summary#image_stats#channels#max"],
                    "output_classes": len(data_cfg["stats_summary#label_stats#labels"]),
                }
            )
        modality = data_src_cfg.get("modality", "ct").lower()
        spacing = data_cfg["stats_summary#image_stats#spacing#median"]
        spacing[-1] = -1.0

        intensity_upper_bound = float(data_cfg["stats_summary#image_foreground_stats#intensity#percentile_99_5"])
        intensity_lower_bound = float(data_cfg["stats_summary#image_foreground_stats#intensity#percentile_00_5"])
        ct_intensity_xform = {
            "_target_": "Compose",
            "transforms": [
                {
                    "_target_": "ScaleIntensityRanged",
                    "keys": "@image_key",
                    "a_min": intensity_lower_bound,
                    "a_max": intensity_upper_bound,
                    "b_min": 0.0,
                    "b_max": 1.0,
                    "clip": True,
                },
                {"_target_": "CropForegroundd", "keys": ["@image_key", "@label_key"], "source_key": "@image_key"},
            ],
        }
        mr_intensity_transform = {
            "_target_": "NormalizeIntensityd",
            "keys": "@image_key",
            "nonzero": True,
            "channel_wise": True,
        }
        for key in ["transforms_infer", "transforms_train", "transforms_validate"]:
            for idx, xform in enumerate(self.cfg[f"{key}#transforms"]):
                if isinstance(xform, dict) and xform.get("_target_", "").startswith("Spacing"):
                    xform["pixdim"] = deepcopy(spacing)
                elif isinstance(xform, str) and xform.startswith("PLACEHOLDER_INTENSITY_NORMALIZATION"):
                    if modality.startswith("ct"):
                        self.cfg[f"{key}#transforms#{idx}"] = deepcopy(ct_intensity_xform)
                    else:
                        self.cfg[f"{key}#transforms#{idx}"] = deepcopy(mr_intensity_transform)


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire({"Segresnet2DAlgo": Segresnet2DAlgo})

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
import glob
import inspect
import os
import shutil

import yaml

from monai.apps.auto3dseg.algorithms.algorithm_configurator import AlgorithmConfigurator


class Configurator(AlgorithmConfigurator):
    def __init__(self, data_stats_filename: str = None, input_filename: str = None, output_path: str = None):
        AlgorithmConfigurator.__init__(self, data_stats_filename, input_filename, output_path)

    def load(self):
        with open(self.data_stats_filename) as f:
            self.data_stats = yaml.full_load(f)

        with open(self.input_filename) as f:
            self.input = yaml.full_load(f)

        self.source_path = os.path.dirname(inspect.getfile(self.__class__))
        config_filenames = glob.glob(os.path.join(self.source_path, "configs", "*.yaml"))

        self.config = {}
        for _i in range(len(config_filenames)):
            config_filename = config_filenames[_i]
            _key = os.path.basename(config_filename)

            with open(config_filename) as f:
                self.config[_key] = yaml.full_load(f)

    def update(self):
        patch_size = [320, 320]
        max_shape = self.data_stats["stats_summary"]["image_stats"]["shape"]["max"][0]
        for _k in range(2):
            patch_size[_k] = max(32, max_shape[_k] // 32 * 32) if max_shape[_k] < patch_size[_k] else patch_size[_k]
        modality = self.input["modality"].lower()
        spacing = self.data_stats["stats_summary"]["image_stats"]["spacing"]["median"][0]
        spacing[-1] = -1.0

        for _key in ["hyper_parameters.yaml", "hyper_parameters_search.yaml"]:
            self.config[_key]["bundle_root"] = os.path.join(self.output_path, "dints")
            self.config[_key]["data_file_base_dir"] = self.input["dataroot"]
            self.config[_key]["data_list_file_path"] = self.input["datalist"]
            self.config[_key]["input_channels"] = int(
                self.data_stats["stats_summary"]["image_stats"]["channels"]["max"]
            )
            self.config[_key]["output_classes"] = len(self.data_stats["stats_summary"]["label_stats"]["labels"])

            for _j in range(2):
                self.config[_key]["patch_size"] = patch_size[_j]
                self.config[_key]["patch_size_valid"] = patch_size[_j]

        for _key in ["transforms_infer.yaml", "transforms_train.yaml", "transforms_validate.yaml"]:
            _t_key = [_item for _item in self.config[_key].keys() if "transforms" in _item][0]

            _i_intensity = -1
            for _i in range(len(self.config[_key][_t_key]["transforms"])):
                _t = self.config[_key][_t_key]["transforms"][_i]

                if type(_t) is dict and _t["_target_"] == "Spacingd":
                    _t["pixdim"] = spacing
                elif type(_t) is str and _t == "PLACEHOLDER_INTENSITY_NORMALIZATION":
                    _i_intensity = _i

                self.config[_key][_t_key]["transforms"][_i] = _t

            _t_intensity = []
            if "ct" in modality:
                intensity_upper_bound = float(
                    self.data_stats["stats_summary"]["image_foreground_stats"]["intensity"]["percentile_99_5"][0]
                )
                intensity_lower_bound = float(
                    self.data_stats["stats_summary"]["image_foreground_stats"]["intensity"]["percentile_00_5"][0]
                )
                _t_intensity.append(
                    {
                        "_target_": "ScaleIntensityRanged",
                        "keys": "@image_key",
                        "a_min": intensity_lower_bound,
                        "a_max": intensity_upper_bound,
                        "b_min": 0.0,
                        "b_max": 1.0,
                        "clip": True,
                    }
                )
                _t_intensity.append(
                    {"_target_": "CropForegroundd", "keys": ["@image_key", "@label_key"], "source_key": "@image_key"}
                )
            elif "mr" in modality:
                _t_intensity.append(
                    {"_target_": "NormalizeIntensityd", "keys": "@image_key", "nonzero": True, "channel_wise": True}
                )

            self.config[_key][_t_key]["transforms"] = (
                self.config[_key][_t_key]["transforms"][:_i_intensity]
                + _t_intensity
                + self.config[_key][_t_key]["transforms"][(_i_intensity + 1) :]
            )

    def write(self):
        write_path = os.path.join(self.output_path, "dints")
        if not os.path.exists(write_path):
            os.makedirs(write_path, exist_ok=True)

        if os.path.exists(os.path.join(write_path, "scripts")):
            shutil.rmtree(os.path.join(write_path, "scripts"))

        shutil.copytree(os.path.join(self.source_path, "scripts"), os.path.join(write_path, "scripts"))

        if os.path.exists(os.path.join(write_path, "configs")):
            shutil.rmtree(os.path.join(write_path, "configs"))

        os.makedirs(os.path.join(write_path, "configs"), exist_ok=True)

        for _key in self.config.keys():
            with open(os.path.join(write_path, "configs", _key), "w") as f:
                yaml.dump(self.config[_key], stream=f)

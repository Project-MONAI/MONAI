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

import sys
sys.path.append('/workspace/Code/MONAI/monai/apps/auto3dseg/algorithms/templates')
import copy
import glob
import inspect
import os
import shutil

import yaml

from monai.utils import optional_import
from algorithm_configurator import AlgorithmConfigurator
from segresnet.scripts import roi_ensure_divisible, roi_ensure_levels

np, _ = optional_import("numpy")

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
        patch_size = [224, 224, 144] # default roi
        levels = 5 # default number of hierarchical levels
        resample = False
        image_size = [int(i) for i in self.data_stats['stats_summary']['image_stats']['shape']['percentile_99_5'][0]]
        output_classes = len(self.data_stats["stats_summary"]["label_stats"]["labels"])
        modality = self.input["modality"].lower()
        full_range = [self.data_stats['stats_summary']['image_stats']['intensity']['percentile_00_5'][0], \
                            self.data_stats['stats_summary']['image_stats']['intensity']['percentile_99_5'][0]]
        intensity_lower_bound = float(
            self.data_stats["stats_summary"]["image_foreground_stats"]["intensity"]["percentile_00_5"][0]
        )
        intensity_upper_bound = float(
            self.data_stats["stats_summary"]["image_foreground_stats"]["intensity"]["percentile_99_5"][0]
        )
        spacing_lower_bound = np.array(
            self.data_stats["stats_summary"]["image_stats"]["spacing"]["percentile_00_5"][0]
        )
        spacing_upper_bound = np.array(
            self.data_stats["stats_summary"]["image_stats"]["spacing"]["percentile_99_5"][0]
        )

        # adjust to image size
        patch_size = [min(r, i) for r,i in zip(patch_size, image_size)]  # min for each of spatial dims
        patch_size = roi_ensure_divisible(patch_size, levels = levels)
        # reduce number of levels to smaller then 5 (default) if image is too small
        levels, patch_size = roi_ensure_levels(levels, patch_size, image_size)

        if levels>=5: #default
            num_blocks = [1,2,2,4,4]
        elif levels==4:
            num_blocks = [1,2,2,4]
        elif levels==3:
            num_blocks = [1,3,4]
        elif levels==2:
            num_blocks = [2,6]
        elif levels==1:
            num_blocks = [8]
        else:
            raise ValueError('Strange number of levels'+str(levels))

        for _key in ["hyper_parameters.yaml"]:
            self.config[_key]["bundle_root"] = os.path.join(self.output_path, "segresnet")
            self.config[_key]["data_file_base_dir"] = self.input["dataroot"]
            self.config[_key]["data_list_file_path"] = self.input["datalist"]
            self.config[_key]["input_channels"] = int(
                self.data_stats["stats_summary"]["image_stats"]["channels"]["max"]
            )
            self.config[_key]["output_classes"] = output_classes

            self.config[_key]["patch_size"] = patch_size
            self.config[_key]["patch_size_valid"] = copy.deepcopy(patch_size)

            if 'class_names' in self.input and isinstance(self.input['class_names'], list):
                if isinstance(self.input['class_names'][0], str):
                    self.config[_key]['class_names'] = self.input['class_names']
                else:
                    self.config[_key]['class_names'] = [x['name'] for x in self.input['class_names']]
                    self.config[_key]['class_index'] = [x['index'] for x in self.input['class_names']]
                    #check for overlap
                    all_ind = []
                    for a in self.config[_key]['class_index']:
                        if bool(set(all_ind) & set(a)): # overlap found
                            self.config[_key]['softmax'] = False
                            break
                        all_ind = all_ind + a

            if "ct" in modality:
                spacing = [1.0, 1.0, 1.0]

                #make sure intensity range is a valid CT range
                is_valid_for_ct = full_range[0] < -300 and full_range[1] > 300
                if is_valid_for_ct:
                    lrange = intensity_upper_bound-intensity_lower_bound
                    if lrange < 500: #make sure range is at least 500 points
                        intensity_lower_bound -= (500-lrange)//2
                        intensity_upper_bound += (500-lrange)//2
                    intensity_lower_bound = max(intensity_lower_bound, -1250) # limit to -1250..1500
                    intensity_upper_bound = min(intensity_upper_bound, 1500)

            elif "mr" in modality:
                spacing = self.data_stats["stats_summary"]["image_stats"]["spacing"]['median'][0]

            self.config[_key]["resolution"] = spacing # resample on the fly to this resolution
            self.config[_key]['intensity_bounds'] = [intensity_lower_bound, intensity_upper_bound]

            if np.any(spacing_lower_bound/np.array(spacing) < 0.5) or np.any(spacing_upper_bound/np.array(spacing) > 1.5):
                # Resampling recommended to median resolution
                resample = True
                self.config[_key]['resample'] = resample

            n_cases = len(self.data_stats['stats_by_cases'])
            max_epochs = int(np.clip( np.ceil(80000.0 / n_cases), a_min=300, a_max=1250))
            warmup_epochs = int(np.ceil(0.01 * max_epochs))

            self.config[_key]['num_epochs'] = max_epochs

            if 'warmup_epochs' in self.config[_key]['lr_scheduler']:
                self.config[_key]['lr_scheduler']['warmup_epochs'] = warmup_epochs


        for _key in ["network.yaml"]:
            self.config[_key]["network"]["blocks_down"] = num_blocks
            self.config[_key]["network"]["blocks_up"] = [1]* (len(num_blocks)-1) # [1,1,1,1..]
            if self.input['multigpu']:
                self.config[_key]["network"]['norm'] = ['BATCH', {'affine': True}] # use batchnorm with multi gpu
            else:
                self.config[_key]["network"]['norm'] = ["INSTANCE", {"affine": True}] # use instancenorm with single gpu

        for _key in ["transforms_train.yaml", "transforms_validate.yaml"]:
            _t_key = [_item for _item in self.config[_key].keys() if "transforms" in _item][0]
            if 'class_index' not in self.config["hyper_parameters.yaml"] or not isinstance(self.config["hyper_parameters.yaml"]['class_index'], list):
                pass
            else:
                self.config[_key][_t_key]["transforms"].append(
                    {
                        "_target_": "LabelMapping",
                        "keys": "@label_key",
                        "class_index": self.config["hyper_parameters.yaml"]['class_index'],
                    }
                )

        for _key in ["transforms_train.yaml"]:
            # get crop transform
            _t_crop = []
            should_crop_based_on_foreground = any([r < 0.5*i for r,i in zip(patch_size, image_size) ]) # if any patch_size less tehn 0.5*image size
            if should_crop_based_on_foreground:
                # Image is much larger then patch_size, using foreground cropping
                ratios = None # equal sampling
                _t_crop.append(
                    {
                        "_target_": "RandCropByLabelClassesd",
                        "keys": ["@image_key", "@label_key"],
                        "label_key": "@label_key",
                        "spatial_size": patch_size,
                        "num_classes": output_classes,
                        "num_samples": 1,
                        "ratios": ratios
                    }
                )
            else:
                # Image size is only slightly larger then patch_size, using random cropping
                _t_crop.append(
                    {
                        "_target_": "RandSpatialCropd",
                        "keys": ["@image_key", "@label_key"],
                        "roi_size": patch_size,
                        "random_size": False
                    }
                )

            _i_crop = -1
            for _i in range(len(self.config[_key]["transforms_train"]["transforms"])):
                _t = self.config[_key]["transforms_train"]["transforms"][_i]

                if type(_t) is str and _t == "PLACEHOLDER_CROP":
                    _i_crop = _i

                self.config[_key]["transforms_train"]["transforms"][_i] = _t

            self.config[_key]["transforms_train"]["transforms"] = (
                self.config[_key]["transforms_train"]["transforms"][:_i_crop]
                + _t_crop
                + self.config[_key]["transforms_train"]["transforms"][(_i_crop + 1) :]
            )


        for _key in ["transforms_infer.yaml", "transforms_train.yaml", "transforms_validate.yaml"]:
            # get intensity transform
            _t_intensity = []
            if "ct" in modality:
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
                if "infer" in _key:
                    _t_intensity.append(
                        {"_target_": "CropForegroundd", "keys": "@image_key", "source_key": "@image_key"}
                    )
                else:
                    _t_intensity.append(
                        {"_target_": "CropForegroundd", "keys": ["@image_key", "@label_key"], "source_key": "@image_key"}
                    )
            elif "mr" in modality:
                _t_intensity.append(
                    {"_target_": "NormalizeIntensityd", "keys": "@image_key", "nonzero": True, "channel_wise": True}
                )

            _t_key = [_item for _item in self.config[_key].keys() if "transforms" in _item][0]
            for _i in range(len(self.config[_key][_t_key]["transforms"])):
                _t = self.config[_key][_t_key]["transforms"][_i]
                if type(_t) is dict and _t["_target_"] == "Spacingd":
                    if resample:
                        _t["pixdim"] = spacing
                    else:
                        self.config[_key][_t_key]["transforms"].pop(_i)
                    break

            _i_intensity = -1
            for _i in range(len(self.config[_key][_t_key]["transforms"])):
                _t = self.config[_key][_t_key]["transforms"][_i]

                if type(_t) is str and _t == "PLACEHOLDER_INTENSITY_NORMALIZATION":
                    _i_intensity = _i

                self.config[_key][_t_key]["transforms"][_i] = _t

            self.config[_key][_t_key]["transforms"] = (
                self.config[_key][_t_key]["transforms"][:_i_intensity]
                + _t_intensity
                + self.config[_key][_t_key]["transforms"][(_i_intensity + 1) :]
            )


    def write(self):
        write_path = os.path.join(self.output_path, "segresnet")
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

            with open(os.path.join(write_path, "configs", _key), "r+") as f:
                lines = f.readlines()
                f.seek(0)
                f.write("# generated automatically by monai.auto3dseg\n")
                f.write("# for more information please visit: https://docs.monai.io/\n\n")
                f.writelines(lines)

if __name__ == '__main__':
    configurator = Configurator(data_stats_filename='/workspace/Code/Bundles/Task05/datastats.yaml', \
                                input_filename='/workspace/Code/Bundles/Task05/input.yaml', \
                                output_path='/workspace/Code/Bundles/Task05')
    configurator.load()
    configurator.update()
    configurator.write()

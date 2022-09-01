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

from monai.utils import optional_import
from monai.apps.auto3dseg import BundleAlgo
from monai.bundle import ConfigParser
from segresnet.scripts import roi_ensure_divisible, roi_ensure_levels

np, _ = optional_import("numpy")

class SegresnetAlgo(BundleAlgo):
    def fill_template_config(self, data_stats):
        if data_stats is None:
            return
        data_cfg = ConfigParser(globals=False)
        if os.path.exists(str(data_stats)):
            data_cfg.read_config(str(data_stats))
        else:
            data_cfg.update(data_stats)
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
        patch_size = [224, 224, 144] # default roi
        levels = 5 # default number of hierarchical levels
        resample = False
        image_size = [int(i) for i in data_cfg['stats_summary']['image_stats']['shape']['percentile_99_5'][0]]
        output_classes = len(data_cfg["stats_summary"]["label_stats"]["labels"])
        modality = data_src_cfg["modality"].lower()
        full_range = [data_cfg['stats_summary']['image_stats']['intensity']['percentile_00_5'][0], \
                            data_cfg['stats_summary']['image_stats']['intensity']['percentile_99_5'][0]]
        intensity_lower_bound = float(
            data_cfg["stats_summary"]["image_foreground_stats"]["intensity"]["percentile_00_5"][0]
        )
        intensity_upper_bound = float(
            data_cfg["stats_summary"]["image_foreground_stats"]["intensity"]["percentile_99_5"][0]
        )
        spacing_lower_bound = np.array(
            data_cfg["stats_summary"]["image_stats"]["spacing"]["percentile_00_5"][0]
        )
        spacing_upper_bound = np.array(
            data_cfg["stats_summary"]["image_stats"]["spacing"]["percentile_99_5"][0]
        )

        # adjust to image size
        patch_size = [min(r, i) for r,i in zip(patch_size, image_size)]  # min for each of spatial dims
        patch_size = roi_ensure_divisible(patch_size, levels = levels)
        # reduce number of levels to smaller then 5 (default) if image is too small
        levels, patch_size = roi_ensure_levels(levels, patch_size, image_size)

        self.cfg["patch_size"] = patch_size
        self.cfg["patch_size_valid"] = deepcopy(patch_size)

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

        # update network config
        self.cfg["network"]["blocks_down"] = num_blocks
        self.cfg["network"]["blocks_up"] = [1]* (len(num_blocks)-1) # [1,1,1,1..]
        if data_src_cfg['multigpu']:
            self.cfg["network"]['norm'] = ['BATCH', {'affine': True}] # use batchnorm with multi gpu
            self.cfg["network"]['act'] = ['RELU', {'inplace': False}] # set act to be not in-place with multi gpu
        else:
            self.cfg["network"]['norm'] = ["INSTANCE", {"affine": True}] # use instancenorm with single gpu

        # update hyper_parameters config 
        if 'class_names' in data_src_cfg and isinstance(data_src_cfg['class_names'], list):
            if isinstance(data_src_cfg['class_names'][0], str):
                self.cfg['class_names'] = data_src_cfg['class_names']
            else:
                self.cfg['class_names'] = [x['name'] for x in data_src_cfg['class_names']]
                self.cfg['class_index'] = [x['index'] for x in data_src_cfg['class_names']]
                #check for overlap
                all_ind = []
                for a in self.cfg['class_index']:
                    if bool(set(all_ind) & set(a)): # overlap found
                        self.cfg['softmax'] = False
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
            spacing = data_cfg["stats_summary"]["image_stats"]["spacing"]['median'][0]

        self.cfg["resolution"] = spacing # resample on the fly to this resolution
        self.cfg['intensity_bounds'] = [intensity_lower_bound, intensity_upper_bound]

        if np.any(spacing_lower_bound/np.array(spacing) < 0.5) or np.any(spacing_upper_bound/np.array(spacing) > 1.5):
            # Resampling recommended to median resolution
            resample = True
            self.cfg['resample'] = resample

        n_cases = len(data_cfg['stats_by_cases'])
        max_epochs = int(np.clip( np.ceil(80000.0 / n_cases), a_min=300, a_max=1250))
        warmup_epochs = int(np.ceil(0.01 * max_epochs))

        self.cfg['num_epochs'] = max_epochs
        if 'warmup_epochs' in self.cfg['lr_scheduler']:
            self.cfg['lr_scheduler']['warmup_epochs'] = warmup_epochs

        # update transform config
        for key in ["transforms_train", "transforms_validate"]:
            if 'class_index' not in self.cfg or not isinstance(self.cfg['class_index'], list):
                pass
            else:
                self.cfg[key]["transforms"].append(
                    {
                        "_target_": "LabelMapping",
                        "keys": "@label_key",
                        "class_index": self.cfg['class_index'],
                    }
                )

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
                    "roi_size": deepcopy(patch_size),
                    "random_size": False
                }
            )

        _i_crop = -1
        for _i in range(len(self.cfg["transforms_train"]["transforms"])):
            _t = self.cfg["transforms_train"]["transforms"][_i]

            if type(_t) is str and _t == "PLACEHOLDER_CROP":
                _i_crop = _i
            self.cfg["transforms_train"]["transforms"][_i] = _t

        self.cfg["transforms_train"]["transforms"] = (
            self.cfg["transforms_train"]["transforms"][:_i_crop]
            + _t_crop
            + self.cfg["transforms_train"]["transforms"][(_i_crop + 1) :]
        )


        for key in ["transforms_infer", "transforms_train", "transforms_validate"]:
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
                if "infer" in key:
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

            for _i in range(len(self.cfg[key]["transforms"])):
                _t = self.cfg[key]["transforms"][_i]
                if type(_t) is dict and _t["_target_"] == "Spacingd":
                    if resample:
                        _t["pixdim"] = spacing
                    else:
                        self.cfg[key]["transforms"].pop(_i)
                    break

            _i_intensity = -1
            for _i in range(len(self.cfg[key]["transforms"])):
                _t = self.cfg[key]["transforms"][_i]

                if type(_t) is str and _t == "PLACEHOLDER_INTENSITY_NORMALIZATION":
                    _i_intensity = _i

                self.cfg[key]["transforms"][_i] = _t

            self.cfg[key]["transforms"] = (
                self.cfg[key]["transforms"][:_i_intensity]
                + _t_intensity
                + self.cfg[key]["transforms"][(_i_intensity + 1) :]
            )

if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire({"SegresnetAlgo": SegresnetAlgo})

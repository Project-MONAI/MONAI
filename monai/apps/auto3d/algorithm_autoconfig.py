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

"""
Step 2 of the Auto3D pipeline. The algorithms are automatically curated and configured in this step.
"""

import argparse
import shutil
from glob import glob
from os import getcwd, makedirs, path
from typing import Dict, List, Union

from monai.apps.utils import get_logger
from monai.bundle.config_parser import ConfigParser

logger = get_logger(module_name=__name__)

__all__ = ["auto_configer", "AutoConfigerUNet"]


def auto_configer(net: str, **args):
    """
    Automatically configure a set of algorithms and copy training and inference scripts
    to the working directory

    Args:
        net:    a string representing the name of network. It is case-insensitive
    Returns:
        AutoConfiger object that associates with the right class
    Raise:
        NotImplementedError if the network class is not implemented in Auto3D
    """

    supported_algorithms = ["UNet"]
    for algo in supported_algorithms:
        if algo.lower() == net.lower():
            configer_class = eval("AutoConfiger" + algo)
            return configer_class(**args)

    raise NotImplementedError


class AutoConfigerBase:
    """
    The AutoConfigerBase is a base class for algorithm auto-configuration. It

    Args:
        datastat: a comprehensive dictionary that has the statistics of images, label and other info
        datalist: a Python dictionary storing group, fold, and other information of the medical
            image dataset, or a string to the JSON file storing the dictionary
        dataroot: user's local directory containing the datasets
        class_names: name of the classes to use in the labeled datasets for training/validation
        name: name of the task, for example: spleen, lung, etc
        task: type of the problem to solve, for example: segmentation
        modality: medical imaging modality, for example: mri, ct
        multigpu: the hardware platform supports utilization of multiple GPUs

    """

    def __init__(
        self,
        datastat: Union[str, Dict, None] = None,
        datalist: Union[str, Dict, None] = None,
        dataroot: Union[str, None] = None,
        class_names: Union[List[str], None] = None,
        name: Union[str, None] = None,
        task: str = "segmentation",
        modality: Union[str, None] = None,
        multigpu: bool = False,
    ):
        if isinstance(datastat, str):
            if not path.isfile(datastat):
                raise ValueError("datastat is not Found: " + datastat)
            self._datastat = ConfigParser.load_config_file(datastat)
        else:
            self._datastat = datastat

        self._method_name = None
        self._datastat = datastat
        self._name = name
        self._datalist = datalist
        self._dataroot = dataroot
        self._class_names = class_names
        self._task = task
        self._modality = modality
        self._multigpu = multigpu

    def configure(self):
        """
        a rule-based process to select algorithms and configure network hyper-params

        Returns:
            a dictionary that records the hyper-parameters from configuration

        """

        config = self.configure_init()

        config = self.configure_net_io(config)

        config = self.configure_train_strategy(config)

        config = self.configure_transforms_modality(config)

        self.config = config

    def configure_init(self):
        """
        configure the input and output channels for the neural network based on data statistics
        initialize a dictionary with required values

        Returns:
            a dictionary that records the hyper-parameters from configuration
        """
        config = dict()

        if self._datastat is None:
            raise ValueError("unknown datastat")

        if self._datalist is None:
            raise ValueError("unknown datalist")
            # todo(mingxin): guess datalist from dataroot
            # todo(mingxin): add this error to documentation

        if self._dataroot is None:
            raise ValueError("unknown dataroot")
            # todo(mingxin): guess dataroot from datalist
            # todo(mingxin): add this error to documentation

        config["datastat"] = self._datastat
        config["datalist"] = self._datalist
        config["dataroot"] = self._dataroot
        
        # name
        if self._name is None:
            from datetime import datetime

            config["name"] = "temp" + datetime.now().strftime("%Y%m%d_%H%M%S")
            print("Input name is not specified, a temporary name is generated (" + config["name"] + ")")
        else:
            config["name"] = self._name

        # task
        if self._task is None:
            config["task"] = "segmentation"
            print("Input task is not specified, assuming segmentation")
        else:
            config["task"] = self._task

        # modality
        if self._modality is None:
            config["modality"] = "mri"
            print("Input modality is not specified, assuming MRI")
        else:
            self._modality = self._modality.lower()

        if self._modality not in ["mri", "ct"]:
            raise ValueError(
                "This modality is currently not supported, only mri or ct, provided:" + str(self._modality)
            )

        config["modality"] = self._modality

        # multi-GPU
        if self._multigpu is None:
            print("Input: multigpu is not specified, assuming False")
            config["multigpu"] = False
        else:
            config["multigpu"] = self._multigpu

        # initialize other required components
        config["net"] = {}
        config["transform"] = {}
        config["train"] = {}

        return config

    def configure_net_io(self, config):
        """
        configure the input and output channels for the neural network based on data statistics
        """
        config["net"]["input_channels"] = int(self._datastat["stats_summary"]["image_stats"]["channels"]["max"])
        config["net"]["output_classes"] = len(self._datastat["stats_summary"]["label_stats"]["labels"])

        max_shape = self._datastat["stats_summary"]["image_stats"]["shape"]["max"][0]
        patch_size = [128, 128, 128]
        patch_size_valid = patch_size

        for _k in range(3):
            patch_size[_k] = max(32, max_shape[_k] // 32 * 32) if max_shape[_k] < patch_size[_k] else patch_size[_k]
        patch_size_valid = patch_size

        config["net"]["patch_size"] = patch_size
        config["net"]["patch_size_valid"] = patch_size_valid

        # todo(mingxin): log the auto-configuration decision process"
        return config

    def configure_train_strategy(self, config):
        """
        todo(mingxin)
        """
        if config['name'] == 'simulate':
            config['train']['num_epoch'] = 10
        elif config['name'] == 'UnitTest':
            config['train']['num_epoch'] = 2
        else:
            config['train']['num_epoch'] = 80

        return config

    def configure_transforms_modality(self, config):
        """
        configure the needed transforms for data augmentations in training/validation
        """
        spacing = [1.0, 1.0, 1.0]
        if config["modality"] == "ct":
            config["transform"]["type"] = "ScaleIntensityRanged+CropForegroundd"

            if self._datastat["stats_summary"]["image_stats"]["spacing"]["median"][0][2] > 3.0:
                spacing = self._datastat["stats_summary"]["image_stats"]["spacing"]["median"]

            config["transform"]["intensity_upper_bound"] = float(
                self._datastat["stats_summary"]["image_foreground_stats"]["intensity"]["percentile_99_5"][0]
            )

            config["transform"]["intensity_lower_bound"] = float(
                self._datastat["stats_summary"]["image_foreground_stats"]["intensity"]["percentile_00_5"][0]
            )

            # todo(mingxin): print the auto-configuration decision process"
        else:
            config["transform"]["type"] = "NormalizeIntensityd"
            spacing = self._datastat["stats_summary"]["image_stats"]["spacing"]["median"]

        config["transform"]["spacing"] = spacing

        return config

    def generate_scripts(self, dest_dir: str = None):
        """
        Generate a configuration file which can be consumed in the following training step
        copy the training and inference scripts to current working directory

        Args:
            dest_dir: a path to put the autoconfig training/inference scripts
        Returns:
            a list of paths pointing to the files generated by this module
        """

        if self._method_name is not None:
            self.source_dir = path.join(path.dirname(path.abspath(__file__)), "algorithms", self._method_name)
            if not dest_dir:
                dest_dir = getcwd()
            script_path = path.join(dest_dir, "autoconfig_" + self._method_name)
            self.config["script_path"] = script_path

            if not path.isdir(script_path):
                makedirs(script_path)  # consider trace clean-up
            for filename in glob(path.join(self.source_dir, "*.*")):
                shutil.copy(filename, script_path)  # consider trace clean-up

            yaml_fpath = path.join(script_path, "auto_config.yaml")
            ConfigParser.export_config_file(self.config, yaml_fpath, fmt="yaml")

        return self.config


class AutoConfigerUNet(AutoConfigerBase):
    """
    The AutoConfigerUNet is an implementation of the AutoConfigurater for the UNet(3D) network

    """

    def __init__(self, **args):
        super().__init__(**args)
        self._method_name = "unet"
        self.configure()


def parse_input(input_yaml: Dict):
    """
    Parse the user arguments and ensure the format when the function is called in a command.

    Args:
        input_yaml: a dictionary populated by yaml

    Return:
        the dictionary in correct format
    """

    # check all defaults
    if "name" not in input_yaml:
        input_yaml["name"] = None

    if "task" not in input_yaml:
        input_yaml["task"] = None

    if "modality" not in input_yaml:
        input_yaml["modality"] = None

    if "multigpu" not in input_yaml:
        input_yaml["multigpu"] = None

    if "datalist" not in input_yaml:
        input_yaml["datalist"] = None

    if "dataroot" not in input_yaml:
        input_yaml["dataroot"] = None

    if "class_names" not in input_yaml:
        input_yaml["class_names"] = None

    return input_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input")
    parser.add_argument("--input", type=str, required=True, help="input info of the dataset")
    parser.add_argument("--output_yaml", type=str, required=False, help="datastat, statistics of the dataset")

    args = parser.parse_args()

    yaml_args = ConfigParser.load_config_file(args.input)
    yaml_args = parse_input(yaml_args)

    networks = ["UNet"]
    for net in networks:
        configer = auto_configer(
            net,
            datastat=yaml_args["datastat"],
            datalist=yaml_args["datalist"],
            dataroot=yaml_args["dataroot"],
            class_names=yaml_args["class_names"],
            name=yaml_args["name"],
            task=yaml_args["task"],
            modality=yaml_args["modality"],
            multigpu=yaml_args["multigpu"],
        )
        configer.generate_scripts()

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
import shutil
from copy import copy, deepcopy
from glob import glob
from typing import Sequence, Union

from monai.auto3dseg.algo_gen import Algo, AlgoGen
from monai.bundle.config_parser import ConfigParser
from monai.utils import ensure_tuple

__all__ = ["BundleAlgo", "BundleGen"]


class BundleAlgo(Algo):
    """
    An algorithm represented by a set of bundle configurations and scripts.

    ``BundleAlgo.cfg`` is a ``monai.bundle.ConfigParser`` instance.

    ..code-block:: python

        from monai.apps.auto3dseg import BundleAlgo

        data_stats_yaml = "/workspace/data_stats.yaml"
        algo = BundleAlgo(
            template_configs=../algorithms/templates/segresnet2d/configs,
            scripts_path="../algorithms/templates/segresnet2d/scripts")
        algo.set_data_stats(data_stats_yaml)
        # algo.set_data_list("../data_list.json")
        algo.export_to_disk(".", algo_name="segresnet2d_1")

    """

    def __init__(self, template_configs=None, scripts_path=None, meta_data_filename=None, parser_args=None):
        if os.path.isdir(template_configs):
            self.template_configs = []
            for ext in ("json", "yaml"):
                self.template_configs += glob(os.path.join(template_configs, f"*.{ext}"))
        else:
            self.template_configs = template_configs
        self.meta_data_filename = meta_data_filename
        self.cfg = ConfigParser(globals=False)  # TODO: define root folder (variable)?
        if self.template_configs is not None:
            self.load_templates(self.template_configs, meta_data_filename, parser_args)

        self.scripts_path = scripts_path
        self.data_stats_files = None
        self.data_list_file = None
        self.output_path = None
        self.name = None

    def load_templates(self, config_files, metadata_file=None, parser_args=None):
        parser_args = parser_args or {}
        self.cfg.read_config(config_files, **parser_args)
        if metadata_file is not None:
            self.cfg.read_meta(metadata_file)

    def set_data_stats(self, data_stats_files):
        self.data_stats_files = data_stats_files

    def set_data_source(self, data_list_file):
        self.data_list_file = data_list_file

    def fill_template_config(self, data_stats_filename):
        # self.algorithm_dir = os.path.join(self.output_path, "segresnet2d")
        # self.inference_script = "scripts/infer.py"
        pass

    def export_to_disk(self, output_path, algo_name, **kwargs):
        self.fill_template_config(self.data_stats_files)
        write_path = os.path.join(output_path, algo_name)
        self.cfg["bundle_root"] = write_path
        os.makedirs(write_path, exist_ok=True)
        # handling scripts files
        output_scripts_path = os.path.join(write_path, "scripts")
        if os.path.exists(output_scripts_path):
            shutil.rmtree(output_scripts_path)
        if self.scripts_path is not None and os.path.exists(self.scripts_path):
            shutil.copytree(self.scripts_path, output_scripts_path)
        # handling config files
        output_config_path = os.path.join(write_path, "configs")
        if os.path.exists(output_config_path):
            shutil.rmtree(output_config_path)
        os.makedirs(output_config_path, exist_ok=True)
        output_config_file = os.path.join(output_config_path, "algo_config.yaml")
        ConfigParser.export_config_file(self.cfg.config, output_config_file, fmt="yaml", default_flow_style=None)
        with open(output_config_file, "r+") as f:
            lines = f.readlines()
            f.seek(0)
            f.write(f"# Generated automatically by `{__name__}`\n")
            f.write("# For more information please visit: https://docs.monai.io/\n\n")
            for item in ensure_tuple(self.template_configs):
                f.write(f"# source file: {item}\n")
            f.write("\n\n")
            f.writelines(lines)
        print(write_path)
        self.output_path = write_path

    def train(self):
        pass

    def get_score(self, *args, **kwargs):
        pass


default_algos = {
    "segresnet2d": dict(
        _target_="monai.apps.auto3dseg.algo_templates.segresnet2d.scripts.algo.Segresnet2DAlgo",
        template_configs="../algorithms/templates/segresnet2d/configs",
        scripts_path="../algorithms/templates/segresnet2d/scripts",
    ),
    "dints": dict(
        _target_="monai.apps.auto3dseg.algo_templates.dints.scripts.algo.DintsAlgo",
        template_configs="../algorithms/templates/dints/configs",
        scripts_path="../algorithms/templates/dints/scripts",
    ),
}


class BundleGen(AlgoGen):
    """
    This class generates a set of bundles according to the cross-validation folds, each of them can run independently.

    .. code-block:: bash

        python -m monai.apps.auto3dseg BundleGen generate --data_stats_filename="../algorithms/data_stats.yaml"

    """

    def __init__(self, algos=None, data_stats_filename=None, data_lists_filename=None):
        self.algos = []
        if algos is None:
            algos = copy(default_algos)
        if isinstance(algos, dict):
            for algo_name, algo_params in algos.items():
                self.algos.append(ConfigParser(algo_params).get_parsed_content())
                self.algos[-1].name = algo_name
        else:
            self.algos = ensure_tuple(algos)

        self.data_stats_filename = data_stats_filename
        self.data_lists_filename = data_lists_filename
        self.history = []

    def set_data_stats(self, data_stats_filename):
        self.data_stats_filename = data_stats_filename

    def get_data_stats(self, fold_idx=0):
        return self.data_stats_filename

    def set_data_lists(self, data_lists_filename):
        self.data_lists_filename = data_lists_filename

    def get_data_lists(self, fold_idx=0):
        return self.data_lists_filename

    def get_history(self, *args, **kwargs):
        return self.history

    def generate(self, fold_idx: Union[int, Sequence[int]] = 0):
        for algo in self.algos:
            for f_id in ensure_tuple(fold_idx):
                data_stats = self.get_data_stats(fold_idx)
                data_lists = self.get_data_lists(fold_idx)
                algo.set_data_stats(data_stats)
                algo.set_data_source(data_lists)
                name = f"{algo.name}_{f_id}"
                algo.export_to_disk(".", algo_name=name)
                self.history.append({name: deepcopy(algo)})  # track the previous, may create a persistent history

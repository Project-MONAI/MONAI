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
import yaml
from monai.bundle.config_parser import ConfigParser
from monai.auto3dseg.algo_generator import Algo, AlgoGen


class BundleAlgo(Algo):
    def __init__(self, template_configs=None, scripts_path=None, meta_data_filename: str=None, parser_args=None):
        self.template_configs = template_configs
        self.meta_data_filename = meta_data_filename
        self.cfg = self.load_templates(template_configs, parser_args)

        self.scripts_path = scripts_path
        self.data_stats_filename = None

    def load_templates(self, configs, parser_args=None):
        cfg = ConfigParser()
        cfg.read_config(configs, **parser_args)
        if self.meta_data_filename is not None:
            cfg.read_meta(self.meta_data_filename)
        return cfg

    def set_data_stats(self, data_stats_filename):
        self.data_stats_filename = data_stats_filename
        self.cfg = fill_template(self.cfg, data_stats)

    def export_to_disk(self, output_path, name, **kwargs):
        write_path = os.path.join(output_path, name)
        if not os.path.exists(write_path):
            os.makedirs(write_path, exist_ok=True)

        if os.path.exists(os.path.join(write_path, "scripts")):
            shutil.rmtree(os.path.join(write_path, "scripts"))

        shutil.copytree(os.path.join(self.scripts_path, "scripts"), os.path.join(write_path, "scripts"))

        if os.path.exists(os.path.join(write_path, "configs")):
            shutil.rmtree(os.path.join(write_path, "configs"))

        os.makedirs(os.path.join(write_path, "configs"), exist_ok=True)

        for _key in self.cfg.keys():
            with open(os.path.join(write_path, "configs", _key), "w") as f:
                yaml.dump(self.cfg[_key], stream=f)

            with open(os.path.join(write_path, "configs", _key), "r+") as f:
                lines = f.readlines()
                f.seek(0)
                f.write("# generated automatically by monai.auto3dseg\n")
                f.write("# for more information please visit: https://docs.monai.io/\n\n")
                f.writelines(lines)

    def get_commands(self, mode="train"):
        pass

    def train(self):
        cmd = self.get_commands(mode="train")
        # launch commands


class BundleGen(AlgoGen):
    def set_candidates(self, *args, **kwargs):
        self.candidates = None

    def generate(self, fold_idx: int = 0):
        for algo in self.candidates:
            data_stats = self.get_data_stats()
            algo = BundleAlgo(algo)
            algo.set_data_stats(data_stats)
            algo.save_to_disk()

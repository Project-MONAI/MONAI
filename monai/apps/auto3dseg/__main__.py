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

from monai.apps.auto3dseg.auto_runner import AutoRunner
from monai.apps.auto3dseg.bundle_gen import BundleAlgo, BundleGen
from monai.apps.auto3dseg.data_analyzer import DataAnalyzer
from monai.apps.auto3dseg.ensemble_builder import AlgoEnsembleBuilder
from monai.apps.auto3dseg.hpo_gen import NNIGen, OptunaGen

if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire(
        {
            "DataAnalyzer": DataAnalyzer,
            "BundleGen": BundleGen,
            "BundleAlgo": BundleAlgo,
            "AlgoEnsembleBuilder": AlgoEnsembleBuilder,
            "AutoRunner": AutoRunner,
            "NNIGen": NNIGen,
            "OptunaGen": OptunaGen,
        }
    )

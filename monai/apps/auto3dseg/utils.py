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
from typing import Dict, List, Optional

from monai.apps.auto3dseg.bundle_gen import BundleAlgo
from monai.auto3dseg import algo_from_pickle, algo_to_pickle


def import_bundle_algo_history(
    output_folder: str = ".", template_path: Optional[str] = None, only_trained: bool = True
) -> List:
    """
    import the history of the bundleAlgo object with their names/identifiers

    Args:
        output_folder: the root path of the algorithms templates.
        template_path: the algorithm_template. It must contain algo.py in the follow path:
            ``{algorithm_templates_dir}/{network}/scripts/algo.py``.
        only_trained: only read the algo history if the algo is trained.
    """

    history = []

    for name in os.listdir(output_folder):
        write_path = os.path.join(output_folder, name)

        if not os.path.isdir(write_path):
            continue

        obj_filename = os.path.join(write_path, "algo_object.pkl")
        if not os.path.isfile(obj_filename):  # saved mode pkl
            continue

        algo, algo_meta_data = algo_from_pickle(obj_filename, template_path=template_path)

        if isinstance(algo, BundleAlgo):  # algo's template path needs override
            algo.template_path = algo_meta_data["template_path"]

        if only_trained:
            if "best_metrics" in algo_meta_data:
                history.append({name: algo})
        else:
            history.append({name: algo})

    return history


def export_bundle_algo_history(history: List[Dict[str, BundleAlgo]]):
    """
    Save all the BundleAlgo in the history to algo_object.pkl in each individual folder

    Args:
        history: a List of Bundle. Typicall the history can be obtained from BundleGen get_history method
    """
    for task in history:
        for _, algo in task.items():
            algo_to_pickle(algo, template_path=algo.template_path)

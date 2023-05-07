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

from __future__ import annotations

import os

from monai.apps.auto3dseg.bundle_gen import BundleAlgo
from monai.auto3dseg import algo_from_pickle, algo_to_pickle
from monai.utils.enums import AlgoKeys

__all__ = ["import_bundle_algo_history", "export_bundle_algo_history", "get_name_from_algo_id"]


def import_bundle_algo_history(
    output_folder: str = ".", template_path: str | None = None, only_trained: bool = True
) -> list:
    """
    import the history of the bundleAlgo objects as a list of algo dicts.
    each algo_dict has keys name (folder name), algo (bundleAlgo), is_trained (bool),

    Args:
        output_folder: the root path of the algorithms templates.
        template_path: the algorithm_template. It must contain algo.py in the follow path:
            ``{algorithm_templates_dir}/{network}/scripts/algo.py``.
        only_trained: only read the algo history if the algo is trained.
    """

    history = []

    for name in sorted(os.listdir(output_folder)):
        write_path = os.path.join(output_folder, name)

        if not os.path.isdir(write_path):
            continue

        obj_filename = os.path.join(write_path, "algo_object.pkl")
        if not os.path.isfile(obj_filename):  # saved mode pkl
            continue

        algo, algo_meta_data = algo_from_pickle(obj_filename, template_path=template_path)

        best_metric = algo_meta_data.get(AlgoKeys.SCORE, None)
        if best_metric is None:
            try:
                best_metric = algo.get_score()
            except BaseException:
                pass

        is_trained = best_metric is not None

        if (only_trained and is_trained) or not only_trained:
            history.append(
                {AlgoKeys.ID: name, AlgoKeys.ALGO: algo, AlgoKeys.SCORE: best_metric, AlgoKeys.IS_TRAINED: is_trained}
            )

    return history


def export_bundle_algo_history(history: list[dict[str, BundleAlgo]]) -> None:
    """
    Save all the BundleAlgo in the history to algo_object.pkl in each individual folder

    Args:
        history: a List of Bundle. Typically, the history can be obtained from BundleGen get_history method
    """
    for algo_dict in history:
        algo = algo_dict[AlgoKeys.ALGO]
        algo_to_pickle(algo, template_path=algo.template_path)


def get_name_from_algo_id(id: str) -> str:
    """
    Get the name of Algo from the identifier of the Algo.

    Args:
        id: identifier which follows a convention of "name_fold_other".

    Returns:
        name of the Algo.
    """
    return id.split("_")[0]

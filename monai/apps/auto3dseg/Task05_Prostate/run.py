#!/usr/bin/env python

import importlib
import json
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.dont_write_bytecode = True

data_stats_filename = "datastats.yaml"
input_filename = "input.yaml"

selected_algorithms = [
    {
        "name": "dints",
        "configurator": "monai.apps.auto3dseg.algorithms.templates.dints.configurator",
    },
    {
        "name": "segresnet2d",
        "configurator": "monai.apps.auto3dseg.algorithms.templates.segresnet2d.configurator",
    },
]

for _i in range(len(selected_algorithms)):
    selected_algorithm = selected_algorithms[_i]

    if os.path.exists(os.path.join(os.getcwd(), selected_algorithm["name"])):
        shutil.rmtree(os.path.join(os.getcwd(), selected_algorithm["name"]))

    module = importlib.import_module(selected_algorithm["configurator"])
    class_ = getattr(module, "Configurator")
    configurator = class_(
        data_stats_filename=data_stats_filename,
        input_filename=input_filename,
        output_path=os.getcwd(),
    )

    configurator.load()
    configurator.update()
    configurator.write()

    selected_algorithm["algorithm_dir"] = configurator.algorithm_dir
    selected_algorithm["inference_script"] = configurator.inference_script

    print(json.dumps(selected_algorithm, indent=4))

    module = importlib.import_module(
        "monai.apps.auto3dseg.algorithms.templates.{:s}.algo".format(
            selected_algorithm["name"]
        )
    )
    class_ = getattr(module, "Algo")
    algo = class_(algorithm_dir=selected_algorithm["algorithm_dir"])

    algo.update(
        {
            "num_iterations": 50,
            "num_iterations_per_validation": 25,
        }
    )
    best_metric = algo.train()
    print("best_metric", best_metric)

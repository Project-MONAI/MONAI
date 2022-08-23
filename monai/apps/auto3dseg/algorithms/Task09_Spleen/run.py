#!/usr/bin/env python

import importlib
import monai
import os
import sys

sys.dont_write_bytecode = True

data_stats_filename = "monai/apps/auto3dseg/algorithms/Task09_Spleen/datastats.yaml"
input_filename = "monai/apps/auto3dseg/algorithms/Task09_Spleen/input.yaml"

selected_algorithm = {
    "name": "swinUNETR",
    "configurator": "monai.apps.auto3dseg.algorithms.templates.swinUNETR.configurator",
}

module = importlib.import_module(selected_algorithm["configurator"])
class_ = getattr(module, "Configurator")

configurator = class_(
    data_stats_filename = data_stats_filename,
    input_filename = input_filename,
    output_path = 'monai/apps/auto3dseg/algorithms/templates',
)

configurator.load()
configurator.update()
configurator.write()

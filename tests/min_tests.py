# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import sys
import unittest


def run_testsuit():
    exclude_cases = [  # these cases use external dependencies
        "test_arraydataset",
        "test_cachedataset",
        "test_cachedataset_parallel",
        "test_check_md5",
        "test_dataset",
        "test_ahnet",
        "test_handler_checkpoint_loader",
        "test_handler_checkpoint_saver",
        "test_handler_classification_saver",
        "test_handler_lr_scheduler",
        "test_handler_mean_dice",
        "test_handler_rocauc",
        "test_handler_segmentation_saver",
        "test_handler_stats",
        "test_handler_tb_image",
        "test_handler_tb_stats",
        "test_handler_validation",
        "test_header_correct",
        "test_img2tensorboard",
        "test_integration_segmentation_3d",
        "test_integration_sliding_window",
        "test_integration_unet_2d",
        "test_integration_workflows",
        "test_integration_workflows_gan",
        "test_keep_largest_connected_component",
        "test_keep_largest_connected_componentd",
        "test_load_nifti",
        "test_load_niftid",
        "test_load_png",
        "test_load_pngd",
        "test_load_spacing_orientation",
        "test_nifti_dataset",
        "test_nifti_header_revise",
        "test_nifti_rw",
        "test_nifti_saver",
        "test_orientation",
        "test_orientationd",
        "test_parallel_execution",
        "test_persistentdataset",
        "test_plot_2d_or_3d_image",
        "test_png_rw",
        "test_png_saver",
        "test_rand_rotate",
        "test_rand_rotated",
        "test_rand_zoom",
        "test_rand_zoomd",
        "test_resize",
        "test_resized",
        "test_rotate",
        "test_rotated",
        "test_spacing",
        "test_spacingd",
        "test_zoom",
        "test_zoom_affine",
        "test_zoomd",
        "test_load_image",
        "test_load_imaged",
        "test_smartcachedataset",
        "test_lltm",
    ]

    files = glob.glob(os.path.join(os.path.dirname(__file__), "test_*.py"))

    cases = []
    for case in files:
        test_module = os.path.basename(case)[:-3]
        if test_module in exclude_cases:
            print(f"skipping test {test_module}.")
        else:
            cases.append(f"tests.{test_module}")
    test_suite = unittest.TestLoader().loadTestsFromNames(cases)
    return test_suite


if __name__ == "__main__":
    test_runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = test_runner.run(run_testsuit())
    exit(int(not result.wasSuccessful()))

#!/bin/bash

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

set -e
# script for running the examples


# install necessary packages
pip install numpy
pip install torch
pip install 'monai[itk, nibabel, pillow]'


# home directory
homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEMP_LOG="temp.txt"

cd "$homedir"
find "$homedir" -type f -name $TEMP_LOG -delete


# download data to specific directory
if [ -e "./testing_ixi_t1.tar.gz" ] && [ -d "./workspace/" ]; then
	echo "1" >> $TEMP_LOG
else
	wget  https://www.dropbox.com/s/y890gb6axzzqff5/testing_ixi_t1.tar.gz?dl=1
        mv testing_ixi_t1.tar.gz?dl=1 testing_ixi_t1.tar.gz
        mkdir -p ./workspace/data/medical/ixi/IXI-T1/
        tar -C ./workspace/data/medical/ixi/IXI-T1/ -xf testing_ixi_t1.tar.gz
fi


# run training files in examples/classification_3d
for file in "examples/classification_3d"/*train*
do
    python "$file"
done

# check training files generated from examples/classification_3d
[ -e "./best_metric_model_classification3d_array.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples classification_3d: model file not generated" | tee $TEMP_LOG && exit 0)
[ -e "./best_metric_model_classification3d_dict.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples classification_3d: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval files in examples/classification_3d
for file in "examples/classification_3d"/*eval*
do
    python "$file"
done


# run training files in examples/classification_3d_ignite
for file in "examples/classification_3d_ignite"/*train*
do
    python "$file"
done

# check training files generated from examples/classification_3d_ignite
[ -e "./runs_array/net_checkpoint_20.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples classification_3d_ignite: model file not generated" | tee $TEMP_LOG && exit 0)
[ -e "./runs_dict/net_checkpoint_20.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples classification_3d_ignite: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval files in examples/classification_3d_ignite
for file in "examples/classification_3d_ignite"/*eval*
do
    python "$file"
done


# run training files in examples/segmentation_2d
for file in "examples/segmentation_2d"/*train*
do
    python "$file"
done

# check training files generated from examples/segmentation_2d
[ -e "./best_metric_model_segmentation2d_array.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples segmentation_2d: model file not generated" | tee $TEMP_LOG && exit 0)
[ -e "./best_metric_model_segmentation2d_dict.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples segmentation_2d: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval files in examples/segmentation_2d
for file in "examples/segmentation_2d"/*eval*
do
    python "$file"
done


# run training files in examples/segmentation_3d
for file in "examples/segmentation_3d"/*train*
do
    python "$file"
done

# check training files generated from examples/segmentation_3d
[ -e "./best_metric_model_segmentation3d_array.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples segmentation_3d: model file not generated" | tee $TEMP_LOG && exit 0)
[ -e "./best_metric_model_segmentation3d_dict.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples segmentation_3d: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval files in examples/segmentation_3d
for file in "examples/segmentation_3d"/*eval*
do
    python "$file"
done


# run training files in examples/segmentation_3d_ignite
for file in "examples/segmentation_3d_ignite"/*train*
do
    python "$file"
done

# check training files generated from examples/segmentation_3d_ignite
[ -e "./runs_array/net_checkpoint_100.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples segmentation_3d_ignite: model file not generated" | tee $TEMP_LOG && exit 0)
[ -e "./runs_dict/net_checkpoint_50.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples segmentation_3d_ignite: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval files in examples/segmentation_3d_ignite
for file in "examples/segmentation_3d_ignite"/*eval*
do
    python "$file"
done


# run training file in examples/workflows
for file in "examples/workflows"/*train*
do
    python "$file"
done

# check training file generated from examples/workflows
[ -e "./runs/net_key_metric*.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples workflows: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval file in examples/workflows
for file in "examples/workflows"/*eval*
do
    python "$file"
done


# run training file in examples/synthesis
for file in "examples/synthesis"/*train*
do
    python "$file"
done

# check training file generated from examples/synthesis
[ -e "./model_out/*.pth" ] && echo "1" >> $TEMP_LOG || (echo "examples synthesis: model file not generated" | tee $TEMP_LOG && exit 0)

# run eval file in examples/synthesis
for file in "examples/synthesis"/*eval*
do
    python "$file"
done

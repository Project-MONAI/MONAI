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

#!/bin/bash

if python -c "import skimage" &> /dev/null; then
    echo "[info] skimage is installed"
else
    # Update pip
    python -m pip install -U pip
    # Install scikit-image
    python -m pip install -U scikit-image
fi

if python -c "import nibabel" &> /dev/null; then
    echo "[info] nibabel is installed"
else
    pip install nibabel
fi

if python -c "import monai" &> /dev/null; then
    echo "[info] monai is installed"
else
    pip install git+https://github.com/Project-MONAI/MONAI#egg=monai
fi

FOLD=${1}
CONFIG_ALGO="${2}/auto_config.yaml"
NUM_GPUS_PER_NODE=${3}

NUM_NODES=1

if [ ${NUM_GPUS_PER_NODE} -eq 1 ]
then
    export CUDA_VISIBLE_DEVICES=0
elif [ ${NUM_GPUS_PER_NODE} -eq 2 ]
then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ ${NUM_GPUS_PER_NODE} -eq 4 ]
then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
elif [ ${NUM_GPUS_PER_NODE} -eq 8 ]
then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

OUTPUT_ROOT="${2}/models/fold${FOLD}"
JSON_KEY="training"

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=3456 \
    ${2}/train.py   --config_algo=${CONFIG_ALGO} \
                    --fold=${FOLD} \
                    --json_key=${JSON_KEY} \
                    --output_root=${OUTPUT_ROOT} \

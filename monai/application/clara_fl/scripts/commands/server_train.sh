#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/workspace/data/medical/Clara/federated-learning/src"
export MMAR_ROOT="/workspace/data/medical/MONAI/monai/application/clara_fl/scripts"

SERVER_FILE=config/config_fed_server.json

python3 -u -m monai.application.clara_fl.server.fed_aggregate \
    -m $MMAR_ROOT \
    -s $SERVER_FILE \
    --set \
    MMAR_CKPT=$MMAR_ROOT/models/net_key_metric=0.4006.pth \
    MMAR_CKPT_DIR=$MMAR_ROOT/models \
    secure_train=true

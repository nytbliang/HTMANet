#!/usr/bin/env bash

set -x

# file definition
CONFIG_FILE=$1
CONFIG_PY="${CONFIG_FILE##*/}"
CONFIG="${CONFIG_PY%.*}"
WORK_DIR="./work_dirs/${CONFIG}"

# train config
GPUS=1
PORT=${PORT:-29511}
RANDOM_SEED=0
export CUDA_VISIBLE_DEVICES=0 #,1,2,3

if [ ! -d "${WORK_DIR}" ]; then
  mkdir -p "${WORK_DIR}"
  cp "${CONFIG_FILE}" $0 "${WORK_DIR}"
fi

echo -e "\n config file: ${CONFIG}\n"

# training
echo -e '\nDistributed Training\n'
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ./tools/train.py ${CONFIG_FILE} \
    --seed $RANDOM_SEED \
    --launcher 'pytorch' \
    --work-dir $WORK_DIR \
    --resume-from work_dirs/tmanet_r50-d8_769x769_80k_cityscapes_video/iter_100000.pth

echo -e "\nWork Dir: ${WORK_DIR}.\n"
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

llmc=$(cd "$(dirname "$0")/.."; pwd)
export PYTHONPATH=$llmc:$PYTHONPATH

config=${llmc}/configs/quantization/lords/omniq_lords_llama3_8b_w4a16_g128.yml

find_unused_port() {
    while true; do
        port=$(shuf -i 10000-60000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo "$port"
            return 0
        fi
    done
}
UNUSED_PORT=$(find_unused_port)

torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    --rdzv_id $UNUSED_PORT \
    --rdzv_backend c10d \
    --rdzv_endpoint 127.0.0.1:$UNUSED_PORT \
    ${llmc}/llmc/__main__.py --config $config --task_id $UNUSED_PORT

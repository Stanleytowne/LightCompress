#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

llmc=$(cd "$(dirname "$0")/.."; pwd)
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=rtn_llama3_8b_w4a16_g128
config=${llmc}/configs/quantization/lords/rtn_llama3_8b_w4a16_g128.yml

nnodes=1
nproc_per_node=1

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

MASTER_ADDR=127.0.0.1
MASTER_PORT=$UNUSED_PORT
task_id=$UNUSED_PORT

nohup \
torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $task_id \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
${llmc}/llmc/__main__.py --config $config --task_id $task_id \
> ${task_name}.log 2>&1 &

sleep 2
ps aux | grep '__main__.py' | grep $task_id | awk '{print $2}' > ${task_name}.pid

echo "Started ${task_name} (PID file: ${task_name}.pid, Log: ${task_name}.log)"
echo "Kill with: xargs kill -9 < ${task_name}.pid"

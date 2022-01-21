#!/bin/bash

set -e

export DATASETS_ROOT="/mnt/nas/raid0/workspace/datasets/wip"

trap 'jobs -p | xargs kill' SIGINT
trap 'jobs -p | xargs kill' SIGSTOP

for session in {0..9}
do
    echo "============== Task: $session =============="
    for testCase in {0..4}
    do
        CUDA_VISIBLE_DEVICES=0 python cifar.py $((2*testCase)) $session &
        CUDA_VISIBLE_DEVICES=1 python cifar.py $((2*testCase+1)) $session
    done
    
    sleep 5
done
#!/bin/bash

export DATASETS_ROOT="/mnt/nas/raid0/workspace/datasets/wip"

for session in {0..4}
do
    for testCase in {0..4}
    do
        CUDA_VISIBLE_DEVICES=0 python mnist.py $((2*testCase)) $session &
        CUDA_VISIBLE_DEVICES=1 python mnist.py $((2*testCase+1)) $session
    done
    
    sleep 5
done
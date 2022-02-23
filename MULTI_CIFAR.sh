#!/bin/bash

set -e

export DATASETS_ROOT="~/data"

trap 'jobs -p | xargs kill' SIGINT
trap 'jobs -p | xargs kill' SIGSTOP

for session in {0..10}
do
    echo "============== Task: $session =============="
    for testCase in {0..2}
    do
        CUDA_VISIBLE_DEVICES=0 python multi_cifar.py $((4*testCase)) $session &
        CUDA_VISIBLE_DEVICES=1 python multi_cifar.py $((4*testCase+1)) $session &
        CUDA_VISIBLE_DEVICES=2 python multi_cifar.py $((4*testCase+2)) $session &
        CUDA_VISIBLE_DEVICES=3 python multi_cifar.py $((4*testCase+3)) $session
    done

    sleep 5
done
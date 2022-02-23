#!/bin/bash

set -e

export DATASETS_ROOT="~/data"

trap 'jobs -p | xargs kill' SIGINT
trap 'jobs -p | xargs kill' SIGSTOP

for session in {0..10}
do
    echo "============== Task: $session =============="
    for testCase in {0..3}
    do
        CUDA_VISIBLE_DEVICES=testCase python multi_cifar.py $((testCase+4)) $session &
    done

    sleep 5
done
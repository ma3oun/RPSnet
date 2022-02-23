#!/bin/bash

set -e

export DATASETS_ROOT="~/data"

trap 'jobs -p | xargs kill' SIGINT
trap 'jobs -p | xargs kill' SIGSTOP

for session in {0..10}
do
    echo "============== Task: $session =============="
      CUDA_VISIBLE_DEVICES=4 python multi_cifar.py $((4)) $session &
      CUDA_VISIBLE_DEVICES=5 python multi_cifar.py $((5)) $session &
      CUDA_VISIBLE_DEVICES=6 python multi_cifar.py $((6)) $session &
      CUDA_VISIBLE_DEVICES=7 python multi_cifar.py $((7)) $session

    sleep 5
done
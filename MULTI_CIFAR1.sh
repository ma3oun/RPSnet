#!/bin/bash

set -e

export DATASETS_ROOT="~/data"

trap 'jobs -p | xargs kill' SIGINT
trap 'jobs -p | xargs kill' SIGSTOP

for session in {0..10}
do
    echo "============== Task: $session =============="
      CUDA_VISIBLE_DEVICES=0 python multi_cifar.py $((0)) $session &
      CUDA_VISIBLE_DEVICES=1 python multi_cifar.py $((1)) $session &
      CUDA_VISIBLE_DEVICES=2 python multi_cifar.py $((2)) $session &
      CUDA_VISIBLE_DEVICES=3 python multi_cifar.py $((3)) $session

    sleep 5
done
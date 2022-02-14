"""
RPS network training on CIFAR100
Copyright (c) Jathushan Rajasegaran, 2019
"""

import sys
from runner import main
from rps_net import RPS_net_mlp
from datasets import mnistDataset


class args:
    datasetName = "mnist"
    epochs = 10
    checkpoint = "results/mnist/RPS_net_mnist"
    savepoint = "results/mnist/pathnet_mnist"
    dataset = "MNIST"
    num_class = 10
    class_per_task = 2
    M = 8
    nLayers = 2
    N = 1
    lr = 0.001
    batchSize = 128
    workers = 16
    resume = False
    arch = "mlp"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [6, 8, 16]
    gamma = 0.5
    rigidness_coff = 2.5
    jump = 1
    memory = 4400


if __name__ == "__main__":
    state = {
        key: value
        for key, value in args.__dict__.items()
        if not key.startswith("__") and not callable(key)
    }
    print(state)
    model = RPS_net_mlp(args.M)
    print(model)

    current_sess = int(sys.argv[2])
    test_case = sys.argv[1]
    main(args, model, mnistDataset, test_case, current_sess)

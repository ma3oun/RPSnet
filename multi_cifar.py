"""
RPS network training on CIFAR100
Copyright (c) Jathushan Rajasegaran, 2019
"""

import sys
from runner import main
from rps_net import RPS_net_cifar
from datasets import cifar100Dataset, cifar10Dataset

class args:
    datasetName = "multi_cifar"
    checkpoint = "results/multi_cifar/RPS_net_multi_cifar"
    savepoint = ""
    # dataset = "cifar-100"
    M = 8
    nLayers = 9
    N = 1
    batchSize = 128
    workers = 16
    resume = False  # otherwise: indicate path to checkpoint file
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    gamma = 0.5
    rigidness_coff = 2.5
    jump = 2
    memory = 2000

class args_cifar100(args):
    lr = 0.001
    schedule = [20, 40, 60, 80]
    epochs = 100
    num_class = 100
    class_per_task = 10

class args_cifar10(args):
    epochs = 10
    schedule = [0]
    num_class = 10
    class_per_task = 10

if __name__ == "__main__":
    if int(sys.argv[2]) <= 9:
        current_args = args_cifar100
        dataset = cifar100Dataset
    elif int(sys.argv[2]) == 10:
        current_args = args_cifar10
        current_args.lr = args_cifar100.lr
        dataset = cifar10Dataset
    else:
        raise Exception('Session > 10 not expected')

    state = {
        key: value
        for key, value in current_args.__dict__.items()
        if not key.startswith("__") and not callable(key)
    }
    print(state)
    model = RPS_net_cifar(current_args.M)
    print(model)

    current_sess = int(sys.argv[2])
    test_case = sys.argv[1]
    main(current_args, model, dataset, test_case, current_sess)

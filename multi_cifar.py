"""
RPS network training on CIFAR100
Copyright (c) Jathushan Rajasegaran, 2019
"""

import sys
from runner import main
from rps_net import RPS_net_cifar
from datasets import cifar100Dataset, cifar10Dataset
from easydict import EasyDict

params = EasyDict()
params.datasetName = "multi_cifar"
params.checkpoint = "results/multi_cifar/RPS_net_multi_cifar"
params.savepoint = ""
params.schedule = [20, 40, 60, 80]
params.epochs = 100
params.lr = 0.001
params.class_per_task = 10
params.M = 8
params.nLayers = 9
params.N = 1
params.batchSize = 128
params.workers = 16
params.resume = False  # otherwise: indicate path to checkpoint file
params.arch = "res-18"
params.start_epoch = 0
params.evaluate = False
params.sess = 0
params.test_case = 0
params.gamma = 0.5
params.rigidness_coff = 2.5
params.jump = 2
params.memory = 2000
params.with_mlflow = True


if __name__ == "__main__":
    if int(sys.argv[2]) <= 9:
        params.num_class = 100
        dataset = cifar100Dataset
    elif int(sys.argv[2]) == 10:
        params.num_class = 10
        dataset = cifar10Dataset
    else:
        raise Exception("Session > 10 not expected")

    print(params)
    model = RPS_net_cifar(params.M)
    print(model)

    current_sess = int(sys.argv[2])
    test_case = sys.argv[1]
    main(params, model, dataset, test_case, current_sess)

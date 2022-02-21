"""
RPS network training on CIFAR100
Copyright (c) Jathushan Rajasegaran, 2019
"""

import sys
from runner import main
from rps_net import RPS_net_mlp
from datasets import mnistDataset
from easydict import EasyDict


params = EasyDict()


params.datasetName = "mnist"
params.epochs = 10
params.checkpoint = "results/mnist/RPS_net_mnist"
params.savepoint = "results/mnist/pathnet_mnist"
params.dataset = "MNIST"
params.num_class = 10
params.class_per_task = 2
params.M = 8
params.nLayers = 2
params.N = 1
params.lr = 0.001
params.batchSize = 128
params.workers = 16
params.resume = False
params.arch = "mlp"
params.start_epoch = 0
params.evaluate = False
params.sess = 0
params.test_case = 0
params.schedule = [6, 8, 16]
params.gamma = 0.5
params.rigidness_coff = 2.5
params.jump = 1
params.memory = 4400


if __name__ == "__main__":
    print(params)
    model = RPS_net_mlp(params.M)
    print(model)

    current_sess = int(sys.argv[2])
    test_case = sys.argv[1]
    main(params, model, mnistDataset, test_case, current_sess)

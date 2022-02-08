"""
RPS network training on CIFAR100
Copyright (c) Jathushan Rajasegaran, 2019
"""

import os
import time
import random
import torch
import torchvision.transforms as transforms
from utils import mkdir_p

import numpy as np
import copy
import torch
import sys
import random
from torch.utils.data import TensorDataset

from rps_net import RPS_net_cifar
from learner import Learner
from util import *


def load_cifar(
    batchSize: int, currentSession: int, replayBufferSize: int, nClsPerTask: int
):
    from cl_datasets import getDatasets

    trainTransforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    testTransforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    if currentSession == 0:
        train_loader, test_loader = getDatasets(
            f"cifar100_{currentSession}",
            batchSize,
            trainTransforms=trainTransforms,
            testTransforms=testTransforms,
        )
    else:
        # The replay buffer needs to filled
        nElementsPerTask = replayBufferSize // nClsPerTask
        trainDataBuffer = []
        testDataBuffer = []
        for previousSession in range(currentSession):
            previousTrainLoader, previousTestLoader = getDatasets(
                f"cifar100_{previousSession}",
                1,
                trainTransforms=trainTransforms,
                testTransforms=testTransforms,
            )
            trainDataBuffer.extend(
                [(x.squeeze(0), y.squeeze()) for x, y in previousTrainLoader][
                    :nElementsPerTask
                ]
            )
            testDataBuffer.extend(
                [(x.squeeze(0), y.squeeze()) for x, y in previousTestLoader]
            )

        xTensor = torch.stack([x for x, _ in trainDataBuffer])
        yTensor = torch.stack([y for _, y in trainDataBuffer])
        extraTrainingData = TensorDataset(xTensor, yTensor)

        xTensor = torch.stack([x for x, _ in testDataBuffer])
        yTensor = torch.stack([y for _, y in testDataBuffer])
        extraTestData = TensorDataset(xTensor, yTensor)

        train_loader, test_loader = getDatasets(
            f"cifar100_{currentSession}",
            batchSize,
            trainTransforms=trainTransforms,
            testTransforms=testTransforms,
            extraTrainingData=extraTrainingData,
            extraTestData=extraTestData,
        )

    return train_loader, test_loader


class args:
    epochs = 50
    checkpoint = "results/cifar100/RPS_net_cifar"
    savepoint = ""
    dataset = "cifar-100"
    num_class = 100
    class_per_task = 10
    M = 8
    nLayers = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False  # otherwise: indicate path to checkpoint file
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [20, 40, 60, 80]
    gamma = 0.5
    rigidness_coff = 2.5
    jump = 1


state = {
    key: value
    for key, value in args.__dict__.items()
    if not key.startswith("__") and not callable(key)
}
print(state)

# Use CUDA
use_cuda = torch.cuda.is_available()
seed = random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)


def main():
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if not os.path.isdir("models/CIFAR100/" + args.checkpoint.split("/")[-1]):
        mkdir_p("models/CIFAR100/" + args.checkpoint.split("/")[-1])
    args.savepoint = "models/CIFAR100/" + args.checkpoint.split("/")[-1]

    model = RPS_net_cifar(args.M)
    print(model)

    current_sess = int(sys.argv[2])
    test_case = sys.argv[1]
    args.test_case = test_case

    memory = 2000

    train_loader, test_loader = load_cifar(
        args.train_batch, current_sess, memory, args.class_per_task
    )

    if current_sess == 0:
        # First session
        path = np.zeros((args.nLayers, args.M), dtype=bool)
        path[:, 0] = True
        fixed_path = np.zeros((args.nLayers, args.M), dtype=bool)
        train_path = path.copy()
        infer_path = path.copy()
    else:
        load_test_case = get_best_model(current_sess - 1, args.checkpoint)
        if current_sess % args.jump == 0:
            # Get a new path
            fixed_path = np.load(
                os.path.join(
                    args.checkpoint,
                    f"fixed_path_{current_sess -1}_{load_test_case}.npy",
                )
            )

            path = get_path(args.nLayers, args.M, args.N)
            train_path = np.zeros((args.nLayers, args.M), dtype=bool)
        else:
            if current_sess // args.jump == 0:
                # This is the first jump
                fixed_path = np.zeros((args.nLayers, args.M), dtype=bool)
            else:
                # Get data from the last jump
                lastJump = (current_sess // args.jump) * args.jump - 1
                load_test_case_x = get_best_model(lastJump, args.checkpoint)
                fixed_path = np.load(
                    os.path.join(
                        args.checkpoint,
                        f"fixed_path_{lastJump}_{load_test_case_x}.npy",
                    )
                )
            path = np.load(
                os.path.join(
                    args.checkpoint, f"path_{current_sess-1}_{load_test_case}.npy"
                )
            )
            train_path = np.zeros((args.nLayers, args.M), dtype=bool)
        infer_path = np.zeros((args.nLayers, args.M), dtype=bool)
        train_path = ~fixed_path & path
        infer_path = fixed_path | path

    np.save(os.path.join(args.checkpoint, f"path_{current_sess}_{test_case}.npy"), path)
    if current_sess == 0:
        fixed_path_x = path.copy()
    else:
        fixed_path_x = ~fixed_path & path
    np.save(
        os.path.join(args.checkpoint, f"fixed_path_{current_sess}_{test_case}.npy"),
        fixed_path_x,
    )

    print(f"Starting with session {current_sess}")
    print(f"test case : {test_case}")
    print("#" * 80)
    print(f"path\n{path}")
    print(f"fixed_path\n{fixed_path}")
    print(f"train_path\n{train_path}")
    print(f"infer_path\n{infer_path}")

    args.sess = current_sess

    if current_sess > 0:
        path_model = os.path.join(
            args.savepoint,
            f"session_{current_sess - 1}_{load_test_case}_model_best.pth.tar",
        )
        prev_best = torch.load(path_model)
        model.load_state_dict(prev_best["state_dict"])

    main_learner = Learner(
        model=model,
        args=args,
        trainloader=train_loader,
        testloader=test_loader,
        old_model=copy.deepcopy(model),
        use_cuda=use_cuda,
        path=path,
        fixed_path=fixed_path,
        train_path=train_path,
        infer_path=infer_path,
        title="cifar100",
    )
    main_learner.learn()

    if current_sess == 0:
        fixed_path = path.copy()
    else:
        fixed_path = ~fixed_path & path

    np.save(
        os.path.join(args.checkpoint, f"fixed_path_{current_sess}_{test_case}.npy"),
        fixed_path,
    )

    cfmat = main_learner.get_confusion_matrix(infer_path)
    np.save(
        os.path.join(
            args.checkpoint, f"confusion_matrix_{current_sess}_{test_case}.npy"
        ),
        cfmat,
    )

    print(f"done with session {current_sess}")
    print("#" * 80)
    while True:
        if is_all_done(current_sess, args.epochs, args.checkpoint):
            break
        else:
            time.sleep(10)


if __name__ == "__main__":
    main()

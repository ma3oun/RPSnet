"""
RPS network training on CIFAR100
Copyright (c) Jathushan Rajasegaran, 2019
"""

import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
from utils import mkdir_p

import numpy as np
import copy
import sys
import random
from torch.utils.data import TensorDataset

from rps_net import RPS_net_mlp
from learner import Learner
from util import *


class args:
    epochs = 10
    checkpoint = "results/mnist/RPS_net_minst"
    savepoint = "results/mnist/pathnet_mnist"
    dataset = "MNIST"
    num_class = 10
    class_per_task = 2
    M = 8
    nLayers = 2
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
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


state = {
    key: value
    for key, value in args.__dict__.items()
    if not key.startswith("__") and not callable(key)
}
print(state)
memory = 4400
# Use CUDA
use_cuda = torch.cuda.is_available()

seed = random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)


def load_mnist(
    batchSize: int, currentSession: int, replayBufferSize: int, nClsPerTask: int
):
    from cl_datasets import getDatasets

    class FlattenTransform(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.flatten = torch.nn.Flatten()

        def forward(self, img: torch.Tensor) -> torch.Tensor:
            return self.flatten(img)

    trainTransforms = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            FlattenTransform(),
        ]
    )

    testTransforms = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            FlattenTransform(),
        ]
    )

    if currentSession == 0:
        train_loader, test_loader = getDatasets(
            f"mnist_{2*currentSession}{2*currentSession+1}",
            batchSize,
            (32, 32),
            trainTransforms=trainTransforms,
            testTransforms=testTransforms,
        )
    else:
        # The replay buffer needs to filled
        nElementsPerTask = replayBufferSize // nClsPerTask
        dataBuffer = []
        for previousSession in range(currentSession):
            previousLoader, _ = getDatasets(
                f"mnist_{2*previousSession}{2*previousSession+1}",
                1,
                (32, 32),
                trainTransforms=trainTransforms,
                testTransforms=testTransforms,
            )
            dataBuffer.extend(
                [(x.squeeze(0), y.squeeze()) for x, y in previousLoader][
                    :nElementsPerTask
                ]
            )

        xTensor = torch.stack([x for x, _ in dataBuffer])
        yTensor = torch.stack([y for _, y in dataBuffer])
        extraDataset = TensorDataset(xTensor, yTensor)

        train_loader, test_loader = getDatasets(
            f"mnist_{2*currentSession}{2*currentSession+1}",
            batchSize,
            (32, 32),
            trainTransforms=trainTransforms,
            testTransforms=testTransforms,
            extraTrainingData=extraDataset,
        )

    return train_loader, test_loader


def main():

    # create checkpoint directory
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create directory to save models at each checkpoints
    if not os.path.isdir("models/mnist/" + args.checkpoint.split("/")[-1]):
        mkdir_p("models/mnist/" + args.checkpoint.split("/")[-1])
    args.savepoint = "models/mnist/" + args.checkpoint.split("/")[-1]

    model = RPS_net_mlp(args.M)
    print(model)

    current_sess = int(sys.argv[2])
    test_case = sys.argv[1]
    args.test_case = test_case

    train_loader, test_loader = load_mnist(
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
        title="pairwise-mnist",
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

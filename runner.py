"""
RPS network training on CIFAR100
Copyright (c) Jathushan Rajasegaran, 2019
"""

import os
import time
import random
import torch

from utils import mkdir_p

import numpy as np
import copy

from learner import Learner
from paths import get_best_model, get_path, is_all_done
from cl_datasets import Cl_dataset, labelStats


def main(
    args, model: torch.nn.Module, dataset: Cl_dataset, test_case: int, current_sess: int
):
    # Use CUDA
    use_cuda = torch.cuda.is_available()
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if not os.path.isdir(
        f"models/{args.datasetName}/" + args.checkpoint.split("/")[-1]
    ):
        mkdir_p(f"models/{args.datasetName}/" + args.checkpoint.split("/")[-1])
    args.savepoint = f"models/{args.datasetName}/" + args.checkpoint.split("/")[-1]

    args.test_case = test_case

    train_loader, test_loader = dataset.getTaskDataloaders(
        current_sess, args.batchSize, args.memory
    )

    print(f"Train labels description:\n{labelStats(train_loader,args.num_class)}")
    print(f"Test labels description:\n{labelStats(test_loader,args.num_class)}")

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
        title=args.datasetName,
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

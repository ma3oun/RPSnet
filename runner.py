"""
RPS network training on CIFAR100
Copyright (c) Jathushan Rajasegaran, 2019
"""

import copy
import os
import random
import time

import numpy as np
import torch

from cl_datasets import Cl_dataset
from learner import Learner
from paths import get_best_model, load_path, generate_paths, is_all_done
from utils import mkdir_p


def main(
    args, model: torch.nn.Module, dataset: Cl_dataset, test_case: int, current_sess: int
):

    if current_sess == 0 and test_case == 0:
        from datetime import datetime

        os.environ["RPS_NET_RUN_PATH"] = "runs/" + datetime.now().strftime(
            "%d-%m-%Y_%H-%M-%S"
        )
    else:
        while True:
            time.sleep(10)
            if "RPS_NET_RUN_PATH" in os.environ:
                break

    args.checkpoint = os.environ["RPS_NET_RUN_PATH"] + args.checkpoint

    if args.with_mlflow:
        import mlflow

        mlflow.start_run(run_name=f"without_duplicate_paths_{current_sess}_{test_case}")

    # Use CUDA
    use_cuda = torch.cuda.is_available()
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    if not os.path.isdir(os.environ["RPS_NET_RUN_PATH"]):
        mkdir_p(os.environ["RPS_NET_RUN_PATH"])

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if not os.path.isdir(args.checkpoint + "/current_paths"):
        mkdir_p(args.checkpoint + "/current_paths")

    if not os.path.isdir(
        os.environ["RPS_NET_RUN_PATH"]
        + f"models/{args.datasetName}/"
        + args.checkpoint.split("/")[-1]
    ):
        mkdir_p(
            os.environ["RPS_NET_RUN_PATH"]
            + f"models/{args.datasetName}/"
            + args.checkpoint.split("/")[-1]
        )
    args.savepoint = (
        os.environ["RPS_NET_RUN_PATH"]
        + f"models/{args.datasetName}/"
        + args.checkpoint.split("/")[-1]
    )

    args.test_case = test_case

    train_loader, test_loader = dataset.getTaskDataloaders(
        0 if current_sess == 10 else current_sess,
        args.batchSize,
        args.memory,  # TODO - use cifar10 instead of cifar10_0 in a better way
    )

    # print(f"Train labels description:\n{labelStats(train_loader,args.num_class)}")
    # print(f"Test labels description:\n{labelStats(test_loader,args.num_class)}")

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

            # path = get_path(args.nLayers, args.M, args.N)
            if test_case == 0:
                print(f"Generating paths for session {current_sess}")
                generate_paths(
                    args.nLayers,
                    args.M,
                    args.N,
                    fixed_path,
                    args.checkpoint,
                    args.max_test_case,
                )
                print("Paths generated")

            path = None
            while path is None:
                time.sleep(10)
                path = load_path(test_case, args.checkpoint)
                print(f"Loading path_{current_sess}_{test_case}")
            print("Path loaded")

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
                    args.checkpoint, f"path_{current_sess-1}_{load_test_case}.npy",
                )
            )
        train_path = ~fixed_path & path
        infer_path = fixed_path | path

    np.save(
        os.path.join(args.checkpoint, f"path_{current_sess}_{test_case}.npy",), path,
    )

    np.save(
        os.path.join(args.checkpoint, f"fixed_path_{current_sess}_{test_case}.npy",),
        train_path,
    )

    print(f"Starting with session {current_sess}")
    print(f"test case : {test_case}")
    print("#" * 80)
    # print(f"path\n{path}")
    # print(f"fixed_path\n{fixed_path}")
    # print(f"train_path\n{train_path}")
    # print(f"infer_path\n{infer_path}")

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

    cfmat = main_learner.get_confusion_matrix(infer_path)
    np.save(
        os.path.join(
            args.checkpoint, f"confusion_matrix_{current_sess}_{test_case}.npy",
        ),
        cfmat,
    )

    if args.with_mlflow:
        mlflow.log_artifact(
            os.path.join(
                args.checkpoint, f"confusion_matrix_{current_sess}_{test_case}.npy",
            )
        )

    # remove all files in current_paths
    if (
        test_case == 0
        and current_sess > 0
        and current_sess % args.jump is args.jump - 1
    ):
        dir = args.checkpoint + "/current_paths"
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

    print(f"done with session {current_sess}")
    print("#" * 80)
    while True:
        if is_all_done(current_sess, args.epochs, args.checkpoint):
            break
        else:
            time.sleep(10)

    if args.with_mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()

import os
import numpy as np
import glob


def get_best_model(task_search: int, checkpoint: str):

    log_files_a = glob.glob(os.path.join(checkpoint, "session_*.txt"))
    log_files_b = []

    for file in log_files_a:
        filename = os.path.basename(file)
        task = filename.split("session_")[1].split("_")[0]
        if task == str(task_search):
            log_files_b.append(filename)

    best_acc = []
    best_acc_b = []
    for file in log_files_b:
        try:
            f = np.loadtxt(os.path.join(checkpoint, file), skiprows=1)
            best_acc.append(max(f[-1, -1], f[-1, -2]))
            best_acc_b.append(int(file.split("_")[2]))
        except:
            pass

    a = np.argmax(best_acc)
    print(best_acc[a], best_acc_b[a])
    return best_acc_b[a]


def is_all_done(task_search, nEpochs, checkpoint):
    log_files_a = glob.glob(os.path.join(checkpoint, "session_*.txt"))
    log_files_b = []

    for file in log_files_a:
        filename = os.path.basename(file)
        task = filename.split("session_")[1].split("_")[0]
        if task == str(task_search):
            log_files_b.append(filename)

    for file in log_files_b:
        f = np.loadtxt(os.path.join(checkpoint, file), skiprows=1)
        if len(f) == 0:
            raise RuntimeError(
                f"{file} is empty. Something bad happened! Check logs for previous errors."
            )
        elif len(f) != nEpochs:
            return False
    return True


def get_path(nLayers: int, M: int, N: int) -> np.array:
    """Activate a maximum number of N modules randomly for each layer.
    There are nLayers layers and each layer contains a maximum of M modules.

    Args:
        nLayers (int): Number of layers
        M (int): Modules per layer
        N (int): Number of modules to activate per layer

    Returns:
        np.array: Matrix of active modules for each layer
    """
    assert N <= M
    path = np.zeros((nLayers, M), dtype=bool)
    for layerIdx in range(nLayers):
        j = 0
        while j < N:
            rand_value = int(np.random.rand() * M)
            if not path[layerIdx, rand_value]:
                path[layerIdx, rand_value] = True
                j += 1
    return path


def get_free_path(fixed_path):
    path = fixed_path.copy() * 0
    c = 0
    for level in fixed_path:
        a = np.where(level == 0)[0]
        if len(a) > 0:
            path[c, a[0]] = 1
        c += 1
    return path


def flatten_list(a):
    return np.concatenate(a).ravel()

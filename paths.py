import os
import numpy as np
import glob


def is_all_done(task_search, q, checkpoint):
    log_files_a = os.listdir(checkpoint + "/")
    log_files_b = []

    for file in log_files_a:
        file_split = file.split(".")
        if file_split[-1] == "txt":
            file_split_2 = file_split[0].split("_")
            if file_split_2[0] == "session" and file_split_2[1] == str(task_search):
                log_files_b.append(file)

    for file in log_files_b:
        f = np.loadtxt(checkpoint + "/" + file, skiprows=1)
        if len(f) != q:
            return False
    return True


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


def generate_paths(
    nLayers: int,
    M: int,
    N: int,
    fixed_path: np.array,
    checkpoint: str,
    max_test_case: int = 8,
):
    paths = []
    for test_case in range(max_test_case):
        new_path = get_path(nLayers, M, N)

        equivalent = True
        while equivalent:
            equivalent = False
            for path in paths:
                if equivalent_path(path, new_path, fixed_path):
                    equivalent = True
                    new_path = get_path(nLayers, M, N)
                    break

        paths.append(new_path)

    for test_case, path in enumerate(paths):
        np.save(os.path.join(checkpoint, f"current_paths/path_{test_case}.npy"), path)


def load_path(test_case: int, checkpoint):
    file_path = os.path.join(checkpoint, f"current_paths/path_{test_case}.npy")

    if not os.path.isfile(file_path):
        return None
    else:
        return np.load(file_path)


def equivalent_path(path1: np.array, path2: np.array, fixed_path: np.array) -> bool:
    """
    Check if two paths are equivalent.

    Works only when there are the same number of activated modules per layer.
    It is the case here, see parameter N in get_path(...)

    Explanation, to check if two paths are equivalent :
     - If (l,m) is fixed and activated in path1, it should be activated in path2
     - Because there are the same number of activated modules per layer,
       we can deduce that each path will activate the same number of never-activated modules per layer.
       Therefore, the same number of equivalent things* is... equivalent.

    * never activated modules are all equivalent, modulo the initialization
    """
    return np.array_equal(fixed_path & path1, fixed_path & path2)


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

import numpy as np
import pandas as pd

def _process(func, path, repeat):
    data = []
    try:
        for i in range(repeat):
            data.append(func(np.loadtxt(path.format(i))))
        return np.mean(data), np.std(data)
    except ValueError as e: # should use method below
        if len(data) != 0:
            print(e)
            return np.nan, np.nan

    try:
        for i in range(repeat):
            data.append(func(open(path.format(i)).readlines()))
        return np.mean(data), np.std(data)
    except Exception as e:
        print(e)
        return np.nan, np.nan

def extract_time(lines):
    # example: 'train' took 253.845810 s
    for line in lines:
        line = line.strip()
        if line.startswith("'train'"):
            return float(line.split(' ')[2])
    print("\033[33mWarning:Could not find training time.\033[0m")  # yellow
    return np.nan

def extract_name(path):
    # example: PDE Class Name: PoissonND
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("PDE Class Name:"):
                return line.split(' ')[3]
    print("\033[33mWarning:Could not find PDE Class Name.\033[0m")  # yellow
    return ""

def summary(path, tasknum, repeat, iters):
    columns = ['pde', 'iter', 'run_time', 'run_time_std', 'train_loss', 'train_loss_std', 'mse', 'mse_std', 'l2rel', 'l2rel_std']
    result = []
    for i in range(tasknum):
        name = extract_name('{}/{}-0/log.txt'.format(path, i))
        try:
            run_time_mean, run_time_std = _process(extract_time, '{}/{}-{{}}/log.txt'.format(path, i), repeat)
            train_loss_mean, train_loss_std = _process(lambda data: data[-1, 1], '{}/{}-{{}}/loss.txt'.format(path, i), repeat)
        except (FileNotFoundError, IOError):
            run_time_mean = run_time_std = np.nan
            train_loss_mean = train_loss_std = np.nan
        try:
            mse_mean, mse_std = _process(lambda data: data[-1, 2], '{}/{}-{{}}/errors.txt'.format(path, i), repeat)
            l2rel_mean, l2rel_std = _process(lambda data: data[-1, 5], '{}/{}-{{}}/errors.txt'.format(path, i), repeat)
        except Exception:
            mse_mean = mse_std = np.nan
            l2rel_mean = l2rel_std = np.nan
        result.append([name, iters[i], run_time_mean, run_time_std, train_loss_mean, train_loss_std, mse_mean, mse_std, l2rel_mean, l2rel_std])

    # save csv
    df = pd.DataFrame(result, columns=columns)
    df.to_csv(f'{path}/result.csv')
import os

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import pandas as pd
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK
from src.pde.burger import Burger1D, Burger2D
from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.pde.heat import HeatDarcy, HeatMultiscale, HeatComplex, HeatLongTime, HeatND
from src.pde.ns import NSEquation_Classic, NSEquation_LidDriven, NSEquation_FourCircles, NSEquation_Long
from src.pde.poisson import PoissonClassic, PoissonBoltzmann2D, Poisson3D, Poisson2DManyArea, PoissonND
from src.pde.wave import WaveEquation1D, WaveHeterogeneous, WaveEquation2D_Long
from src.pde.inverse import PoissonInv, HeatInv


def process(func, path, repeat):
    data = []
    try:
        for i in range(repeat):
            data.append(func(np.loadtxt(path.format(i))))
        return np.mean(data), np.std(data)
    except ValueError:
        # should use method below
        assert len(data) == 0

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


pde_list = \
    [Burger1D, Burger2D] + \
    [PoissonClassic, PoissonBoltzmann2D, Poisson3D, Poisson2DManyArea] + \
    [HeatDarcy, HeatMultiscale, HeatComplex, HeatLongTime] + \
    [NSEquation_LidDriven, NSEquation_FourCircles, NSEquation_Long] + \
    [WaveEquation1D, WaveHeterogeneous, WaveEquation2D_Long] + \
    [GrayScottEquation, KuramotoSivashinskyEquation] + \
    [PoissonND, HeatND] + \
    [PoissonInv, HeatInv]

pde_list = [pde.__name__ for pde in pde_list]

columns = ['pde', 'iter', 'run_time', 'run_time_std', 'train_loss', 'train_loss_std', 'mse', 'mse_std', 'l2rel', 'l2rel_std']
result = []

if __name__ == '__main__':
    print(len(pde_list))
    exp_path = input('Enter exp_path: ')
    n_repeat = int(input('Enter num_repeat: '))
    iter = 20000
    for i, name in enumerate(pde_list):
        try:
            run_time_mean, run_time_std = process(extract_time, '{}/{}-{{}}/log.txt'.format(exp_path, i), n_repeat)
            train_loss_mean, train_loss_std = process(lambda data: data[-1, 1], '{}/{}-{{}}/loss.txt'.format(exp_path, i), n_repeat)
        except (FileNotFoundError, IOError):
            run_time_mean = run_time_std = np.nan
            train_loss_mean = train_loss_std = np.nan
        try:
            mse_mean, mse_std = process(lambda data: data[-1, 2], '{}/{}-{{}}/errors.txt'.format(exp_path, i), n_repeat)
            l2rel_mean, l2rel_std = process(lambda data: data[-1, 4], '{}/{}-{{}}/errors.txt'.format(exp_path, i), n_repeat)
        except Exception:
            mse_mean = mse_std = np.nan
            l2rel_mean = l2rel_std = np.nan
        result.append([name, iter, run_time_mean, run_time_std, train_loss_mean, train_loss_std, mse_mean, mse_std, l2rel_mean, l2rel_std])

    # save csv
    df = pd.DataFrame(result, columns=columns)
    df.to_csv(f'{exp_path}/result.csv')

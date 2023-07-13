import argparse
import time
import os
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"
parser = argparse.ArgumentParser(description='PINNBench trainer')
parser.add_argument('--name', type=str, default="benchmark")
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--seed', type=int, default=None)
command_args = parser.parse_args()

date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
trainer = Trainer(f"{date_str}-{command_args.name}", command_args.device)

import numpy as np
import torch
import deepxde as dde
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK
from src.pde.burger import Burger1D, Burger2D
from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.pde.heat import HeatDarcy, HeatMultiscale, HeatComplex, HeatLongTime
from src.pde.ns import NSEquation_Classic, NSEquation_LidDriven, NSEquation_FourCircles, NSEquation_Long
from src.pde.poisson import PoissonClassic, PoissonBoltzmann2D, Poisson3D, Poisson2DManyArea, PoissonND
from src.pde.wave import WaveEquation1D, WaveHeterogeneous, WaveEquation2D_Long
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback
from src.utils.rar import rar_wrapper

pde_list = \
    [Burger1D, Burger2D] + \
    [KuramotoSivashinskyEquation, GrayScottEquation] + \
    [HeatComplex, HeatDarcy, HeatLongTime, HeatMultiscale] + \
    [NSEquation_FourCircles, NSEquation_LidDriven, NSEquation_Long] + \
    [PoissonClassic, Poisson2DManyArea, Poisson3D, PoissonBoltzmann2D, PoissonND] + \
    [WaveEquation1D, WaveEquation2D_Long, WaveHeterogeneous]

for pde_class in pde_list:

    def get_model():
        pde = pde_class()
        # pde.training_points()
        # pde.use_gepinn()
        net = dde.nn.FNN([pde.input_dim] + 5 * [100] + [pde.output_dim], "tanh", "Glorot normal")
        net = net.float()

        loss_weights = np.ones(pde.num_loss)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        model = pde.create_model(net)
        model.compile(opt, loss_weights=loss_weights)
        # model.train = rar_wrapper(pde, model, conf={'interval': 1000, 'count': 1})
        return model

    trainer.add_task(
        get_model, {
            'iterations': 20000,
            'callbacks': [
                TesterCallback(),
                PlotCallback(log_every=2000, fast=True),
                LossCallback(verbose=True),
            ]
        }
    )

trainer.set_repeat(3)

if __name__ == "__main__":
    seed = command_args.seed
    if seed is not None:
        dde.config.set_random_seed(seed)
    trainer.setup(__file__, seed)
    trainer.train_all()

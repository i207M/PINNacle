import argparse
import time
import os
import re
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"
parser = argparse.ArgumentParser(description='PINNBench trainer')
parser.add_argument('--name', type=str, default="benchmark")
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--hidden-layers', type=str, default="100*5")
parser.add_argument('--loss-weight', type=str, default="")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--iter', type=int, default=20000)
parser.add_argument('--logevery', type=int, default=100)
parser.add_argument('--plotevery', type=int, default=2000)

command_args = parser.parse_args()

date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
trainer = Trainer(f"{date_str}-{command_args.name}", command_args.device)

def parse_hidden_layers():
    layers = []
    for s in re.split(r"[,_-]",command_args.hidden_layers):
        if '*' in s:
            siz, num = s.split('*')
            layers += [int(siz)] * int(num)
        else:
            layers += [int(s)]
    return layers
def parse_loss_weight():
    if command_args.loss_weight == '': return None
    weights = []
    for s in re.split(r"[,_-]",command_args.loss_weight):
        weights.append(float(s))
    return weights

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

# 建议不要修改这个示例文件，请将它复制为 benchmark_xxx.py 并按照你的想法更改
pde_list = \
    [Burger1D, Burger2D] + \
    [KuramotoSivashinskyEquation, GrayScottEquation] + \
    [HeatComplex, HeatDarcy, HeatLongTime, HeatMultiscale] + \
    [NSEquation_FourCircles, NSEquation_LidDriven, NSEquation_Long] + \
    [PoissonClassic, Poisson2DManyArea, Poisson3D, PoissonBoltzmann2D, PoissonND] + \
    [WaveEquation1D, WaveEquation2D_Long, WaveHeterogeneous]

for pde_class in pde_list:

    def get_model_dde():
        pde = pde_class()
        # pde.training_points()
        # pde.use_gepinn()
        net = dde.nn.FNN([pde.input_dim] + parse_hidden_layers() + [pde.output_dim], "tanh", "Glorot normal")
        net = net.float()

        loss_weights = parse_loss_weight()
        if loss_weights is None:
            loss_weights = np.ones(pde.num_loss)
        else:
            loss_weights = np.array(loss_weights)
        opt = torch.optim.Adam(net.parameters(), command_args.lr)
        # opt = MultiAdam(net.parameters(), lr=1e-3, betas=(0.99, 0.99), loss_group_idx=[pde.num_pde])

        model = pde.create_model(net)
        model.compile(opt, loss_weights=loss_weights)
        # model.train(**train_args)
        return model

    def get_model_other():
        model = None
        # create a model object which support .train() method, and param @model_save_path is required
        # create it according to command_args, return the model and wait for being trained. 
        # use trainer.add_task(get_model_other, {training args}) to schedule
        return model

    trainer.add_task(
        get_model_dde, {
            'iterations': command_args.iter,
            "display_every": command_args.logevery,
            'callbacks': [
                TesterCallback(log_every=command_args.logevery),
                PlotCallback(log_every=command_args.plotevery, fast=True),
                LossCallback(verbose=True),
            ]
        }
    )

trainer.set_repeat(5)

if __name__ == "__main__":
    seed = command_args.seed
    if seed is not None:
        dde.config.set_random_seed(seed)
    trainer.setup(__file__, seed)
    trainer.train_all()

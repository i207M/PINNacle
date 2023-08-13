import argparse
import time
import os
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"
parser = argparse.ArgumentParser(description='PINNBench trainer')
parser.add_argument('--name', type=str, default="benchmark")
parser.add_argument('--device', type=str, default="0") # set to "cpu" enables cpu training 
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--hidden-layers', type=str, default="100*5")
parser.add_argument('--loss-weight', type=str, default="")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--iter', type=int, default=20000)
parser.add_argument('--log-every', type=int, default=100)
parser.add_argument('--plot-every', type=int, default=2000)
parser.add_argument('--repeat', type=int, default=1)

command_args = parser.parse_args()

date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
trainer = Trainer(f"{date_str}-{command_args.name}", command_args.device)

import numpy as np
import torch
import deepxde as dde
from src.model.laaf import DNN_GAAF, DNN_LAAF
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK, Adam_LBFGS
from src.pde.burger import Burger1D, Burger2D
from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.pde.heat import HeatDarcy, HeatMultiscale, HeatComplex, HeatLongTime, HeatND
from src.pde.ns import NSEquation_Classic, NSEquation_LidDriven, NSEquation_BackStep, NSEquation_Long
from src.pde.poisson import PoissonClassic, PoissonBoltzmann2D, Poisson3D, Poisson2DManyArea, PoissonND
from src.pde.wave import WaveEquation1D, WaveHeterogeneous, WaveEquation2D_Long
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback

# It is recommended not to modify this example file.
# Please copy it as benchmark_xxx.py and make changes according to your own ideas.
pde_list = \
    [Burger1D, Burger2D] + \
    [PoissonClassic, PoissonBoltzmann2D, Poisson3D, Poisson2DManyArea] + \
    [HeatDarcy, HeatMultiscale, HeatComplex, HeatLongTime] + \
    [NSEquation_LidDriven, NSEquation_BackStep, NSEquation_Long] + \
    [WaveEquation1D, WaveHeterogeneous, WaveEquation2D_Long] + \
    [KuramotoSivashinskyEquation, GrayScottEquation] + \
    [PoissonND, HeatND] + \
    [(Burger2D, {"datapath": "ref/burgers2d_1.dat", "icpath": ("ref/burgers2d_init_u_1.dat", "ref/burgers2d_init_v_1.dat")})] + \
    [(Burger2D, {"datapath": "ref/burgers2d_2.dat", "icpath": ("ref/burgers2d_init_u_2.dat", "ref/burgers2d_init_v_2.dat")})] + \
    [(Burger2D, {"datapath": "ref/burgers2d_3.dat", "icpath": ("ref/burgers2d_init_u_3.dat", "ref/burgers2d_init_v_3.dat")})] + \
    [(Burger2D, {"datapath": "ref/burgers2d_4.dat", "icpath": ("ref/burgers2d_init_u_4.dat", "ref/burgers2d_init_v_4.dat")})] + \
    [(PoissonClassic, {"scale": 2})] + \
    [(PoissonClassic, {"scale": 4})] + \
    [(PoissonClassic, {"scale": 8})] + \
    [(PoissonClassic, {"scale": 16})] + \
    [(HeatMultiscale, {"init_coef": (5 * np.pi, np.pi)})] + \
    [(HeatMultiscale, {"init_coef": (10 * np.pi, np.pi)})] + \
    [(HeatMultiscale, {"init_coef": (40 * np.pi, np.pi)})] + \
    [(NSEquation_LidDriven, {"datapath": "ref/lid_driven_a2.dat", "a": 2})] + \
    [(NSEquation_LidDriven, {"datapath": "ref/lid_driven_a8.dat", "a": 8})] + \
    [(NSEquation_LidDriven, {"datapath": "ref/lid_driven_a16.dat", "a": 16})] + \
    [(NSEquation_LidDriven, {"datapath": "ref/lid_driven_a32.dat", "a": 32})] + \
    [(WaveEquation1D, {"a": 2})] + \
    [(WaveEquation1D, {"a": 6})] + \
    [(WaveEquation1D, {"a": 8})] + \
    [(WaveEquation1D, {"a": 10})] + \
    [(HeatND, {"dim": 4})] + \
    [(HeatND, {"dim": 6})] + \
    [(HeatND, {"dim": 8})] + \
    [(HeatND, {"dim": 10})]

for pde_config in pde_list:

    def get_model_dde():
        if isinstance(pde_config, tuple):
            pde = pde_config[0](**pde_config[1])
        else:
            pde = pde_config()
        
        # pde.training_points()
        # pde.use_gepinn()

        net = dde.nn.FNN([pde.input_dim] + parse_hidden_layers(command_args) + [pde.output_dim], "tanh", "Glorot normal")
        net = net.float()

        loss_weights = parse_loss_weight(command_args)
        if loss_weights is None:
            loss_weights = np.ones(pde.num_loss)
        else:
            loss_weights = np.array(loss_weights)
        opt = torch.optim.Adam(net.parameters(), command_args.lr)
        # opt = MultiAdam(net.parameters(), lr=1e-3, betas=(0.99, 0.99), loss_group_idx=[pde.num_pde])

        model = pde.create_model(net)
        model.compile(opt, loss_weights=loss_weights)
        # the trainer calls model.train(**train_args)
        return model

    def get_model_others():
        model = None
        # create a model object which support .train() method, and param @model_save_path is required
        # create the object based on command_args and return it to be trained
        # schedule the task using trainer.add_task(get_model_other, {training args})
        return model

    trainer.add_task(
        get_model_dde, {
            'iterations': command_args.iter,
            "display_every": command_args.log_every,
            'callbacks': [
                TesterCallback(log_every=command_args.log_every),
                PlotCallback(log_every=command_args.plot_every, fast=True),
                LossCallback(verbose=True),
            ]
        }
    )

trainer.set_repeat(command_args.repeat)

if __name__ == "__main__":
    seed = command_args.seed
    if seed is not None:
        dde.config.set_random_seed(seed)
    trainer.setup(__file__, seed)
    trainer.train_all()
    trainer.summary()
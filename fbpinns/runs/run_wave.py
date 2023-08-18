import sys

import numpy as np

sys.path.insert(0, '.')
import problems
from active_schedulers import AllActiveSchedulerND, PointActiveSchedulerND, LineActiveSchedulerND, PlaneActiveSchedulerND, ManualActiveSchedulerND
from constants import Constants, get_subdomain_xs, get_subdomain_ws
from main import FBPINNTrainer, PINNTrainer
from trainersBase import train_models_multiprocess

sys.path.insert(0, 'pdes')
from wave import WaveEquation1D, Wave2DLong, WaveHetergeneous

boundary_n = (1,)
y_n = (0,1)
plot_lims = (1.1, False)
random = True

def run_PINN():
    c = Constants(
        RUN="bench2_PINN_%s_%sh_%sl_%sb"%(P.name, n_hidden, n_layers, batch_size[0]),
        P=P,
        SUBDOMAIN_XS=subdomain_xs,
        BOUNDARY_N=boundary_n,
        Y_N=y_n,
        N_HIDDEN=n_hidden,
        N_LAYERS=n_layers,
        BATCH_SIZE=batch_size,
        RANDOM=random,
        N_STEPS=n_steps,
        BATCH_SIZE_TEST=batch_size_test,
        PLOT_LIMS=plot_lims,
        TEST_FREQ=1000,
        SUMMARY_FREQ=100,
    )
    PINNTrainer(c).train()

def run_FBPINN():
    grid = "x".join([str(len(sx)-1) for sx in subdomain_xs])
    bdw = "" if boundary_weight==-1 else "_bdw"+str(boundary_weight)
    c = Constants(
        RUN="bench2_%s_%s_%sh_%sl_%sb_%sw_%s"%(grid+bdw, P.name, n_hidden, n_layers, batch_size[0], width, A.name),
        P=P,
        SUBDOMAIN_XS=subdomain_xs,
        SUBDOMAIN_WS=subdomain_ws,
        BOUNDARY_N=boundary_n,
        Y_N=y_n,
        ACTIVE_SCHEDULER=A,
        ACTIVE_SCHEDULER_ARGS=args,
        N_HIDDEN=n_hidden,
        N_LAYERS=n_layers,
        BATCH_SIZE=batch_size,
        BOUNDARY_BATCH_SIZE=boundary_batch_size,
        BOUNDARY_WEIGHT=boundary_weight,
        RANDOM=random,
        N_STEPS=n_steps,
        BATCH_SIZE_TEST=batch_size_test,
        BOUNDARY_BATCH_SIZE_TEST=boundary_batch_size_test,
        PLOT_LIMS=plot_lims,
        TEST_FREQ=1000,
        SUMMARY_FREQ=100,
    )
    FBPINNTrainer(c).train()

name="Wave2DLong"
if name=="WaveHetergeneous":
    P = WaveHetergeneous()
    subdomain_xs = [np.array([-1, 1]), np.array([-1, 1]), np.array([0, 2.5, 5])]
    batch_size = (20, 20, 20)
    batch_size_test = (50, 50, 6)
    boundary_batch_size, boundary_batch_size_test, boundary_weight = -1, -1, -1

    n_steps = 10000
    n_hidden, n_layers = 64, 4
    A, args = AllActiveSchedulerND, ()
    width = 0.6
    subdomain_ws = get_subdomain_ws(subdomain_xs, width)
    run_FBPINN()
elif name=="Wave2DLong":
    P = Wave2DLong()
    subdomain_xs = [np.array([0, 1]), np.array([0, 1]), np.array([0, 25, 50, 75, 100])]
    batch_size = (16, 16, 16)
    batch_size_test = (16, 16, 16) # (100, 100) not enough (May 4th)
    boundary_batch_size = 2048
    boundary_batch_size_test = 2048
    boundary_weight = 100

    n_steps = 10000
    n_hidden, n_layers = 64, 4
    A, args = AllActiveSchedulerND, ()
    width = 0.6
    subdomain_ws = get_subdomain_ws(subdomain_xs, width)
    run_FBPINN()
elif name=="WaveEquation1D":
    P = WaveEquation1D()
    subdomain_xs = [np.array([0, 0.5, 1]), np.array([0, 0.5, 1])]
    batch_size = (200, 200)
    batch_size_test = (210, 210)
    boundary_batch_size = -1
    boundary_batch_size_test = -1
    boundary_weight = -1

    n_steps = 10000
    n_hidden, n_layers = 64, 4 # 64, 4
    A, args = AllActiveSchedulerND, ()
    width = 0.6
    subdomain_ws = get_subdomain_ws(subdomain_xs, width)
    run_FBPINN()
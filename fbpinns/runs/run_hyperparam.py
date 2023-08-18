import sys
import numpy as np
import argparse
import json

sys.path.insert(0, '.')
import problems
from active_schedulers import AllActiveSchedulerND, PointActiveSchedulerND, LineActiveSchedulerND, PlaneActiveSchedulerND, ManualActiveSchedulerND
from constants import Constants, get_subdomain_xs, get_subdomain_ws
import models
from main import FBPINNTrainer

sys.path.insert(0, 'pdes')
from heat import HeatComplex
from poisson import Poisson2D_Classic
from chaotic import GrayScott
from burger import Burgers1D

parser = argparse.ArgumentParser(description='Input testcase name')
parser.add_argument('casename', metavar='S', type=str, help='testcase name')
parser.add_argument('hyperparam', type=str, help='hyperparameter: width or div')
parser.add_argument('part', type=str, help='choose value 0,1 (lower) or value 2,3 (upper) or type 0,1,2,3 for hyperparameter')
args = parser.parse_args()
casename = args.casename
hyperp_name = args.hyperparam
if args.part == "lower":
    hpvalslice = slice(0,2)
elif args.part == "upper":
    hpvalslice = slice(2,4)
else:
    ipart = int(args.part)
    hpvalslice = slice(ipart, ipart+1)

P = globals()[casename]()
print(P.d)
conf_all = json.load(open("runs/run_all_config.json"))
conf_all["chaotic"]["GrayScott"]["ba"] = [20, 20, 21] # to avoid empty segments
conf_flat = dict()
for k,v in conf_all.items():
    for kk,vv in v.items():
        conf_flat[kk] = vv
conf_this = conf_flat[casename]
boundary_batch_size = np.prod(np.array(conf_this["ba"])) // 4
boundary_batch_size_test = np.prod(np.array(conf_this["ba_t"])) // 4
boundary_weight = 100
data_weight = 1
subdomain_xs = [np.linspace(l, r, seg+1) for l,r,seg in zip(P.bbox[::2], P.bbox[1::2], conf_this["div"])]
width = 0.6
subdomain_ws = get_subdomain_ws(subdomain_xs, width)
A, args = AllActiveSchedulerND, ()
n_models = 1
for seg in conf_this["div"]:
    n_models *= seg
n_hidden, n_layers = (64, 4) if n_models <= 5 else (16, 2)
usemodel = models.BiFCN if casename[-3:]=="Inv" else models.FCN

grid = "x".join([str(len(sx)-1) for sx in subdomain_xs])
bdw = ("_bdw"+str(boundary_weight) if hasattr(P, "sample_bd") else "") + ("_dw"+str(data_weight) if hasattr(P, "sample_data") else "")
n_steps = 20000

c = Constants(
    RUN="all2_%s_%s_%sh_%sl_%sb_%sw_%s"%(grid+bdw, P.name, n_hidden, n_layers, conf_this["ba"][0], width, A.name),
    P=P,
    SUBDOMAIN_XS=subdomain_xs,
    SUBDOMAIN_WS=subdomain_ws,
    BOUNDARY_N=(1,),
    Y_N=(0,1),
    ACTIVE_SCHEDULER=A,
    ACTIVE_SCHEDULER_ARGS=args,
    MODEL=usemodel,
    N_HIDDEN=n_hidden,
    N_LAYERS=n_layers,
    BATCH_SIZE=tuple(conf_this["ba"]),
    BOUNDARY_BATCH_SIZE=boundary_batch_size,
    BOUNDARY_WEIGHT=boundary_weight,
    DATALOSS_WEIGHT=data_weight,
    RANDOM=True if casename[-3:]=="Inv" else False,
    N_STEPS=n_steps,
    BATCH_SIZE_TEST=tuple(conf_this["ba_t"]),
    BOUNDARY_BATCH_SIZE_TEST=boundary_batch_size_test,
    PLOT_LIMS=(1.1, False),
    TEST_FREQ=2000,
    SUMMARY_FREQ=100,
)

if hyperp_name == "width":
    for seed in range(0,3):
        for width in [0.2, 0.4, 0.6, 0.8][hpvalslice]:
            c.SEED = seed
            c.P = globals()[casename]() # reconstruct P to avoid "Can't pickle local object"
            c.SUBDOMAIN_WS = get_subdomain_ws(c.SUBDOMAIN_XS, width)
            c.hyperparam_name = "width"
            c.hyperparam_value = width
            c.RUN = "hyp_%s_%s_%sh_%sl_%sb_%sw_%s"%(grid+bdw, P.name, n_hidden, n_layers, conf_this["ba"][0], width, A.name)
            FBPINNTrainer(c).train()
elif hyperp_name == "div":
    divs = {"Burgers1D": [[1, 1], [2, 1], [3, 1], [1, 2]],
            "GrayScott": [[1, 1, 1], [1, 1, 3], [1, 1, 5], [2, 2, 1]],
            "HeatComplex":[[1, 1, 1], [1, 1, 3], [1, 1, 5], [2, 2, 1]],
            "Poisson2D_Classic":[[1, 1], [1, 2], [2, 1], [2, 2]]}
    pdivs = divs[casename]
    for seed in range(0,3):
        for pdiv in pdivs[hpvalslice]:
            c.SEED = seed
            c.P = globals()[casename]() # reconstruct P to avoid "Can't pickle local object"
            c.SUBDOMAIN_XS = [np.linspace(l, r, seg+1) for l,r,seg in zip(P.bbox[::2], P.bbox[1::2], pdiv)]
            c.SUBDOMAIN_WS = get_subdomain_ws(c.SUBDOMAIN_XS, 0.6)
            print(c.SUBDOMAIN_XS)
            print(c.SUBDOMAIN_WS)
            c.hyperparam_name = "div"
            c.hyperparam_value = "-".join(str(_) for _ in pdiv)
            grid = "x".join([str(len(sx)-1) for sx in c.SUBDOMAIN_XS])
            c.RUN = "hyp_%s_%s_%sh_%sl_%sb_%sw_%s"%(grid+bdw, P.name, n_hidden, n_layers, conf_this["ba"][0], width, A.name)
            FBPINNTrainer(c).train()
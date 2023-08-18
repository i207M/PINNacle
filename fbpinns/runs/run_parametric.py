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
from heat import HeatMultiscaleExact
from wave import WaveEquation1D
from ns import LidDrivenFlow
from poisson import Poisson2D_Classic

parser = argparse.ArgumentParser(description='Input testcase name')
parser.add_argument('casename', metavar='S', type=str, help='testcase name')
parser.add_argument('pidx', type=int, help='input index of parameter value (0,1,2,3,4)')
parser.add_argument('seed', type=str, default="all", help='all or 0,1,2')
args = parser.parse_args()
casename = args.casename

if casename == "HeatMultiscaleExact":
    paramval = [5, 10, 20, 40][args.pidx]
    P = HeatMultiscaleExact(); P.nx = paramval
elif casename == "WaveEquation1D":
    paramval = [2, 4, 6, 8, 10][args.pidx]
    P = WaveEquation1D(); P.m2 = paramval
elif casename == "LidDrivenFlow":
    paramval = [2, 4, 8, 16, 32][args.pidx]
    P = LidDrivenFlow(paramval)
elif casename == "Poisson2D_Classic":
    paramval = [1, 2, 4, 8, 16][args.pidx]
    P = Poisson2D_Classic(paramval)

if args.seed == 'all':
    sslice = slice(None)
else:
    seedx = int(args.seed)
    sslice = slice(seedx, seedx+1)

print(P.d)
conf_all = json.load(open("runs/run_all_config.json"))
conf_flat = dict()
for k,v in conf_all.items():
    for kk,vv in v.items():
        conf_flat[kk] = vv
conf_flat["HeatMultiscaleExact"] = conf_flat["HeatMultiscale"]
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
    RUN = "prm_%s_%s_%s_%sh_%sl_%sb_%sw_%s"%(str(paramval), grid+bdw, P.name, n_hidden, n_layers, conf_this["ba"][0], width, A.name),
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
c.parameterized_value = paramval

for seed in range(0,3)[sslice]:
    c.SEED = seed
    FBPINNTrainer(c).train()
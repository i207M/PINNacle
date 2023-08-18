import os
import vpinn
from src.pde.poisson import Poisson3d, Poisson2d, Poisson2d_Many_subdomains
from src.pde.ns import NSEquation_Long, NSEquation_LidDriven
from src.pde.burgers import Burgers1d, Burgers2d
from src.pde.heat import Heat2d_Varying_Source, Heat_Multi_scale
from src.pde.wave import WaveEquation1D
from src.config.default_arg import argContainer


current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
os.chdir(current_script_dir)
    
# argContainer.Burgers1d.epoch_num = 10000
# argContainer.Burgers2d.epoch_num = 100
# argContainer.Poisson2d_Many_subdomains.epoch_num = 100
# argContainer.Heat2d_Varying_Source.epoch_num = 100
# argContainer.Poisson3d.epoch_num = 100
# argContainer.Poisson2d.epoch_num = 100
# argContainer.Heat_Multi_scale.epoch_num = 100
argContainer.NSEquation_LidDriven.epoch_num = 10000
# argContainer.WaveEquation1D.epoch_num = 10000

argContainer.to_json()
# p = Burgers2d(device='cuda:0')
# p = Poisson2d(scale=16, device='cuda:0')
# p = Poisson2d_Many_subdomains(device='cuda:0')
# p = Heat2d_Varying_Source(device='cuda:0')
# p = Burgers2d(device='cuda:0', data_id=4)
# p = Heat_Multi_scale(a=40)
p = NSEquation_LidDriven(a=10, device='cuda:0')
# p = WaveEquation1D(a=2, device='cuda:0')

# p.run(load='Burgers2d[3, 30, 30, 30, 2],Q=10,grid_num=[6, 6, 6],test_fcn=5,coef=[1, 1, 10],epoch=100).pth', plotevery=100)
# p.run(load='Poisson2d[2, 15, 15, 15, 1],Q=10,grid_num=[32, 32],test_fcn=5,coef=[0.4, 0.4, 0.4, 0.4, 0.4],epoch=10000).pth', plotevery=10000)
p.run(plotevery=10000)
result = p.plot()[-1]

# if hasattr(p, 'geom') and issubclass(getattr(p, 'geom'), vpinn.geom):
#     pred_grid = pred_grid.unsqueeze(-2)
#     solution_grid = solution_grid.unsqueeze(-2)
print(vpinn.pdebench_err.metric_func(*result))
print(vpinn.loss.mse_loss(*result))
from .kernel import VPINN2d, VPINN3d
from .plot import plot_from_formula, plot_from_data
from .geom import geom, rec, cube
from .geomtime import timeline, timeplane

class PDE:
    def __init__(self, geom:geom, pde, constrains):
        self.geom = geom
        self.pde = pde
        self.constrains = constrains

    def model(self, layer_size, neuron_type, grid_num, coef, Q:int=10, device='cpu',  test_fcn_num=5, load=None, inverse=False, ref=None, ifsave=True):
        self.coef = coef
        self.layer_size = layer_size
        
        if isinstance(self.geom, timeline) or isinstance(self.geom, rec):
            if isinstance(self.geom, timeline): x1, x2, y1, y2 = (self.geom.x1, self.geom.x2, 0, self.geom.t)
            else: x1, x2, y1, y2 = (self.geom.x1, self.geom.x2, self.geom.y1, self.geom.y2)
            
            self.vpinn = VPINN2d(layer_size, self.pde, self.constrains, grid_num, 
                             neuron_type, area=[x1, x2, y1, y2], Q=Q,  
                             test_fcn_num=test_fcn_num, device=device, 
                             load=load, inverse=inverse,ref=ref)
            
        elif isinstance(self.geom, timeplane) or isinstance(self.geom, cube):
            if isinstance(self.geom, timeplane): x1, x2, y1, y2, z1, z2 = (self.geom.x1, self.geom.x2, self.geom.y1,self.geom.y2, 0, self.geom.t)
            else: x1, x2, y1, y2, z1, z2 = (self.geom.x1, self.geom.x2, self.geom.y1,self.geom.y2, self.geom.z1, self.geom.z2)
            # print(grid_num)
            self.vpinn = VPINN3d(layer_size, self.pde, self.constrains, grid_num, 
                             neuron_type, area=[x1, x2, y1, y2, z1, z2], Q=Q,  
                             test_fcn_num=test_fcn_num, device=device, 
                             load=load, inverse=inverse,ref=ref)
        else:
            raise TypeError('Your geom cannot be converted to orthogonal domain.')

    def train(self, model_name, epoch_num=10000, task_id=0, queue=None, lr=1e-3, logevery=100, plotevery=2000):
        self.model_name = model_name
        self.logevery = logevery
        self.plotevery = plotevery
        self.net_need_plotting, self.net = self.vpinn.train(model_name, epoch_num=epoch_num, coef=self.coef, task_id=task_id, queue=queue, lr=lr, logevery=logevery, plotevery=plotevery)
        return self.net
    
    def plot(self, u_ref=None, data=None, grid_data=None, layers=None, style='scatter', scale=1):
        if u_ref is None and data is None:
            raise ValueError('Invalid input')
        result = []
        for i in range(len(self.net_need_plotting)):
            if u_ref:
                result.append(plot_from_formula(self.net_need_plotting[i], self.geom, u_ref, self.model_name, channel=self.layer_size[-1], layers=layers, style=style, epoch=(i + 1) * self.plotevery))

            else:
                result.append(plot_from_data(self.net_need_plotting[i], self.geom, data, self.model_name, channel=self.layer_size[-1], grid_data=grid_data, layers=layers, style=style, epoch=(i + 1) * self.plotevery, scale=scale))
        
        return result
        



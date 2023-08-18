import vpinn
import json
import inspect
import warnings
from abc import ABC, abstractmethod


class Problem(ABC):
    def __init__(self, device='cpu'):
        self.device = device
        self.ref_data = None
    
    # @abstractmethod
    # def pde(self, X, U):
    #     pass
    def load_ref_data(self, path):
        self.ref_data = path
        
    def run(self, load=None, task_id = 0, queue=None, lr=1e-3, logevery=100, plotevery=2000, args=None):
        if (hasattr(self, 'geom') == False and hasattr(self, 'geomtime') == False):
            raise TypeError('Invalid geom or geomtime.')
        
        if (hasattr(self, 'pde') == False):
            raise TypeError('PDE is not defined.')
        
        if (hasattr(self, 'outdim') == False):
            warnings.warn("Outdim of the PDE equation is not defined, the program views it as 1 defaultly.", UserWarning)
            self.outdim = 1
        
        if args:
            self.arg_map = args.to_json()
        else:
            try:
                with open('src/config/default_arg.json', 'r') as f:
                    self.arg_map = json.load(f)
            except Exception as e:
                print("Exception when analyzing config: ", str(e))
        
        
        self.solver = vpinn.solver.PDE(self.geom if hasattr(self, 'geom') else self.geomtime, self.pde, self.constrain if hasattr(self, 'constrain') else None)
        
        if self.__class__.__name__ in self.arg_map:
            
            # User defined arguments are found.
            func_params = inspect.signature(self.solver.model).parameters
            required_params = {k: self.arg_map[self.__class__.__name__][k] for k in func_params if k in self.arg_map[self.__class__.__name__]}
            if hasattr(self, 'inverse'):
                required_params['inverse'] = getattr(self, 'inverse')
            if hasattr(self, 'u_ref'):
                required_params['ref'] = getattr(self, 'u_ref')
            
            self.solver.model(**required_params, device=self.device, load=load)
            # params = {k: self.arg_map[k] for k in ('layer_size', 'neuron_type', 'grid_num', 'coef', 'Q', 'test_fcn_num', 'inverse', 'ref', 'ifsave')}
            # self.solver.model(**params, device=self.device, load=load)
            self.net = self.solver.train(self.__class__.__name__, epoch_num=self.arg_map[self.__class__.__name__]['epoch_num'], task_id=task_id, queue=queue, lr=lr, logevery=logevery, plotevery=plotevery)
            return self.net
        
        else:
            # Use default arguments.
            self.solver.model(
            layer_size=[int(len(self.bbox) / 2), 20, 20, 20, self.outdim], 
            neuron_type='tanh',
            grid_num=[4] * int(len(self.bbox) / 2),
            coef= [1] * len(self.constrain) if hasattr(self, 'constrain') else 0,
            Q=10,
            device=self.device, 
            test_fcn_num=5, 
            load=load,
            inverse=getattr(self, 'inverse') if hasattr(self, 'inverse') else False,
            ref=getattr(self, 'u_ref') if hasattr(self, 'u_ref') else False,
            ifsave=getattr(self, 'ifsave') if hasattr(self, 'ifsave') else False)

            self.net = self.solver.train(self.__class__.__name__, epoch_num=20000, task_id=task_id, queue=queue, lr=lr, logevery=logevery, plotevery=plotevery)
            return self.net[-1]
        

    def plot(self):
        if self.__class__.__name__ in self.arg_map:
            # User defined arguments are found.
            func_params = inspect.signature(self.solver.plot).parameters
            required_params = {}
            for k in func_params:
                if k in self.arg_map[self.__class__.__name__]:
                    value = self.arg_map[self.__class__.__name__][k]
                    if isinstance(value, str):
                        value = value.replace('self.', '', 1) if value.startswith('self.') else value
                    
                    if isinstance(value, str) and hasattr(self, value):
                        required_params[k] = getattr(self, value)
                    else:
                        required_params[k] = value
            if hasattr(self, 'inverse') and getattr(self, 'inverse') and hasattr(self, 'a_ref'):
                required_params['u_ref'] = getattr(self, 'a_ref')
            elif hasattr(self, 'u_ref'):
                required_params['u_ref'] = getattr(self, 'u_ref')
            
            if self.ref_data:
                required_params['data'] = self.ref_data
            
            if hasattr(self, 'scale'):
                required_params['scale'] = getattr(self, 'scale')
            
            if ('u_ref' in required_params)== False and ('data' in required_params) == False:
                print('No plot argument is given.')
                return
        else :
            raise TypeError('No plot config is defined.')
        return self.solver.plot(**required_params)

    
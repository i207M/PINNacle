import torch
from torch.optim import Optimizer

class Adam_LBFGS(Optimizer):
    
    def __init__(
        self, 
        params, 
        switch_epoch=10000, 
        adam_param={'lr': 1e-3, 'betas': (0.9, 0.999)}, 
        lbfgs_param={'lr': 1, 'max_iter': 20}
    ):
        self.params = list(params)
        self.switch_epoch=switch_epoch
        self.adam = torch.optim.Adam(self.params, **adam_param)
        self.lbfgs = torch.optim.LBFGS(self.params, **lbfgs_param)

        super().__init__(self.params, defaults={})

        self.state['current_step'] = 0
    
    def step(self, closure=None):
        self.state['current_step'] += 1

        if self.state['current_step'] < self.switch_epoch:
            self.adam.step(closure)
        else:
            self.lbfgs.step(closure)
            if self.state['current_step'] == self.switch_epoch:
                print(f"Switch to LBFGS at epoch {self.switch_epoch}")

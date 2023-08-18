#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:32:55 2021

@author: bmoseley
"""

# This module defines standard pytorch NN models

# This module is used by constants.py when defining when defining FBPINN / PINN problems

import torch
import torch.nn as nn

total_params = lambda model: sum(p.numel() for p in model.parameters())


class FCN(nn.Module):
    "Fully connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        
        # define layers

        activation = nn.Tanh
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
        # define helper attributes / methods for analysing computational complexity
        
        d1,d2,h,l = N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS
        self.size =           d1*h + h         + (l-1)*(  h*h + h)        +   h*d2 + d2
        self._single_flop = 2*d1*h + h + 5*h   + (l-1)*(2*h*h + h + 5*h)  + 2*h*d2 + d2# assumes Tanh uses 5 FLOPS
        self.flops = lambda BATCH_SIZE: BATCH_SIZE*self._single_flop
        assert self.size == total_params(self)
        
    def forward(self, x):
                
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        
        return x

class BiFCN(nn.Module):
    """
    Parallel Fully connected network
    Nets: Solution, Parameter
    Input: *, 2 (Paremeter net takes the first two dim of input as input)
    Output: 1, 1
    """
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        assert N_OUTPUT == 2
        # define layers

        activation = nn.Tanh
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        
        self.fce = nn.Linear(N_HIDDEN, 1)

        self.fcs2 = nn.Sequential(*[
                        nn.Linear(2, N_HIDDEN),
                        activation()])
        
        self.fch2 = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        
        self.fce2 = nn.Linear(N_HIDDEN, 1)
        
        # define helper attributes / methods for analysing computational complexity
        
        d1,d2,h,l = N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS
        self.size =           d1*h + h         + (l-1)*(  h*h + h)        +   h*d2 + d2
        self._single_flop = 1 #dummy
        self.flops = lambda BATCH_SIZE: BATCH_SIZE*self._single_flop
        #assert self.size == total_params(self)
        
    def forward(self, x):    
        m1, m2 = self.fcs(x), self.fcs2(x[:,:2])
        mm1, mm2 = self.fch(m1), self.fch2(m2)
        o1, o2 = self.fce(mm1), self.fce2(mm2)
        return torch.concat([o1, o2],dim=-1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:16:38 2021

@author: bmoseley
"""

# This module defines various loss functions

# This module is used by main.py and problems.py

import torch


def l2_loss(a, b):
    "L2 loss function"
    
    return torch.mean((a-b)**2)


def l1_loss(a, b):
    "L1 loss function"
    
    return torch.mean(torch.abs(a-b))

def l2_rel_err(y_true, y_hat):
    """
                        ||y_true - y_hat||_2
    L2 relative error = ---------------------
                            ||y_true||_2
    """
    top = torch.sqrt(torch.mean((y_true - y_hat) ** 2))
    bottom = torch.sqrt(torch.mean(y_true ** 2))
    return top / bottom

def l1_rel_err(y_true, y_hat):
    top = torch.mean(torch.abs(y_true - y_hat))
    bottom = torch.mean(torch.abs(y_true))
    return top / bottom

def max_err(y_true, y_hat):
    return torch.max(torch.abs(y_true - y_hat))

def err_csv(y_true, y_hat):
    #return torch.sqrt(torch.mean(\
    #    (torch.sum(y_true,dim=0) - torch.sum(y_hat,dim=0))**2))
    return torch.abs(torch.mean(y_true - y_hat))

import torch
import torch.nn as nn

def mse_loss(data, name, pred, solution):
    mse = nn.MSELoss()
    return mse(pred, solution)

def l1re_loss(data, name, pred, solution):
    l1re = torch.sum(torch.abs(pred - solution)) / torch.sum(torch.abs(solution))
    return l1re

def l2re_loss(data, name, pred, solution):
    l2re = torch.norm(pred - solution) / torch.norm(solution)
    return l2re

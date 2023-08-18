import torch
import torch.nn as nn

# 均方误差 (MSE) 损失函数
def mse_loss(data, name, pred, solution):
    mse = nn.MSELoss()
    return mse(pred, solution)

# 一阶相对误差 (L1RE) 损失函数
def l1re_loss(data, name, pred, solution):
    l1re = torch.sum(torch.abs(pred - solution)) / torch.sum(torch.abs(solution))
    return l1re

# 二阶相对误差 (L2RE) 损失函数
def l2re_loss(data, name, pred, solution):
    l2re = torch.norm(pred - solution) / torch.norm(solution)
    return l2re

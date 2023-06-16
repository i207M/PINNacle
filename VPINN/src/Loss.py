import torch
import torch.nn as nn

# 均方误差 (MSE) 损失函数
def mse_loss(pred, target):
    mse = nn.MSELoss()
    return mse(pred, target)

# 一阶相对误差 (L1RE) 损失函数
def l1re_loss(pred, target):
    l1re = torch.sum(torch.abs(pred - target)) / torch.sum(torch.abs(target))
    return l1re

# 二阶相对误差 (L2RE) 损失函数
def l2re_loss(pred, target):
    l2re = torch.sqrt(torch.norm(pred - target) / torch.norm(target))
    return l2re

def three_loss(pred, target):
    return (mse_loss(pred, target), l1re_loss(pred, target), l2re_loss(pred, target))
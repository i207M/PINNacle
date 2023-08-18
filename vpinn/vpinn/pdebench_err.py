'''
This file is modified from PDEBench. (https://github.com/pdebench/PDEBench/tree/a3eb903582accdd2aa5d62d7f11c58228fa2fa46)
'''

import torch
import math as mt
import numpy as np
from scipy.interpolate import griddata

# def unravel_tensor(grid_num, raveled_tensor, n_last_time_steps, n_components=1):
#     n_x = grid_num[0]
#     n_y = grid_num[1]
#     n_last_time_steps = grid_num[2]
#     return raveled_tensor.reshape((1, n_x, n_y, n_last_time_steps, n_components))

def unravel_tensor(grid_num, data, pred, solution):
    data = data.float().detach().numpy() if isinstance(data, torch.Tensor) else data
    pred = pred.float().detach().numpy() if isinstance(pred, torch.Tensor) else pred
    solution = solution.float().detach().numpy() if isinstance(solution, torch.Tensor) else solution
    
    selected_place = min(data.shape[0], 10000)
    indices = np.random.choice(data.shape[0], selected_place, replace=False)
    data = data[indices]
    pred = pred[indices]
    solution = solution[indices]
    
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    
    if len(grid_num) == 2:
        x = np.linspace(mins[0], maxs[0], grid_num[0])
        y = np.linspace(mins[1], maxs[1], grid_num[1])
        mesh = np.meshgrid(x, y, indexing='ij')
    elif len(grid_num) == 3:
        x = np.linspace(mins[0], maxs[0], grid_num[0])
        y = np.linspace(mins[1], maxs[1], grid_num[1])
        z = np.linspace(mins[2], maxs[2], grid_num[2])
        mesh = np.meshgrid(x, y, z, indexing='ij')
    else:
        raise ValueError("Invalid grid_num length")
    
    coords = np.vstack([m.ravel() for m in mesh]).T
    
    m = pred.shape[1]
    pred_grid = np.zeros((*grid_num, m))
    solution_grid = np.zeros((*grid_num, m))
    for i in range(m):
        pred_grid[..., i] = griddata(data, pred[:, i], coords, method='linear').reshape(grid_num)
        solution_grid[..., i] = griddata(data, solution[:, i], coords, method='linear').reshape(grid_num)
        
    # substitute all nans with 0
    pred_grid[np.isnan(pred_grid)] = 0
    solution_grid[np.isnan(solution_grid)] = 0
    
    return [torch.from_numpy(pred_grid).unsqueeze(0), torch.from_numpy(solution_grid).unsqueeze(0)]

def err_fft(pred, target, Lx=1., Ly=1., Lz=1., iLow=4, iHigh=12, device='cpu'):
    idxs = target.size()
    if len(idxs) == 4:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    if len(idxs) == 5:
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)
    idxs = target.size()
    nb, nc, nt = idxs[0], idxs[1], idxs[-1]
    
    if len(idxs) == 4:  # 1D
        nx = idxs[2]
        pred_F = torch.fft.rfft(pred, dim=2)
        target_F = torch.fft.rfft(target, dim=2)
        _err_F = torch.sqrt(torch.mean(torch.abs(pred_F - target_F) ** 2, axis=0)) / nx * Lx
    if len(idxs) == 5:  # 2D
        pred_F = torch.fft.fftn(pred, dim=[2, 3])
        target_F = torch.fft.fftn(target, dim=[2, 3])
        nx, ny = idxs[2:4]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2), nt]).to(device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                it = mt.floor(mt.sqrt(i ** 2 + j ** 2))
                if it > min(nx // 2, ny // 2) - 1:
                    continue
                err_F[:, :, it] += _err_F[:, :, i, j]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny) * Lx * Ly
    elif len(idxs) == 6:  # 3D
        pred_F = torch.fft.fftn(pred, dim=[2, 3, 4])
        target_F = torch.fft.fftn(target, dim=[2, 3, 4])
        nx, ny, nz = idxs[2:5]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2, nz // 2), nt]).to(device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                for k in range(nz // 2):
                    it = mt.floor(mt.sqrt(i ** 2 + j ** 2 + k ** 2))
                    if it > min(nx // 2, ny // 2, nz // 2) - 1:
                        continue
                    err_F[:, :, it] += _err_F[:, :, i, j, k]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny * nz) * Lx * Ly * Lz

    err_F = torch.zeros([nc, 3, nt]).to(device)
    err_F[:,0] += torch.mean(_err_F[:,:iLow], dim=1)  # low freq
    err_F[:,1] += torch.mean(_err_F[:,iLow:iHigh], dim=1)  # middle freq
    err_F[:,2] += torch.mean(_err_F[:,iHigh:], dim=1)  # high freq
    return err_F

    
def metric_func(data, name, pred, target, n=32):
    # (coordinate, channel)
    
    err_mean = torch.sqrt(torch.mean((pred - target) ** 2, dim=0))
    err_RMSE = err_mean
    nrm = torch.sqrt(torch.mean(target ** 2, dim=0))
    err_nRMSE = err_mean / nrm
    
    err_CSV = torch.abs(torch.sum(pred, dim=0) - torch.sum(target, dim=0))
    
    err_CSV = err_CSV / target.shape[0]
    # worst case in all the data
    err_Max = torch.abs(pred - target).reshape(-1)
    
    pred_grid, solution_grid = unravel_tensor([n, n] if data.shape[1] == 2 else [n, n, n], data, pred, target)
    
    return torch.mean(err_RMSE, dim=[0]).item(), \
               torch.mean(err_nRMSE[err_nRMSE != float('inf')], dim=[0]).item(), \
               torch.mean(err_CSV, dim=[0]).item(), \
               torch.mean(err_Max, dim=[0]).item(), \
               torch.mean(err_fft(pred_grid, solution_grid), dim=[0, -1])
    
    
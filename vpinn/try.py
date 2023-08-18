
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def interpolate_corrected_v10(values, grid_num):
    x_orig = np.linspace(0, 1, X.shape[0])
    y_orig = np.linspace(0, 1, X.shape[1])
    dimensions_orig = (x_orig, y_orig)
    
    x_target = np.linspace(0, 1, grid_num[0])
    y_target = np.linspace(0, 1, grid_num[1])
    
    interpolators = [RegularGridInterpolator(dimensions_orig, values[:, i].reshape(X.shape), bounds_error=False, fill_value=0) 
                     for i in range(values.shape[1])]
    
    new_values = np.zeros((*grid_num, values.shape[1]))
    
    for idx in np.ndindex(*grid_num):
        point = [x_target[idx[0]], y_target[idx[1]]]
        for i in range(values.shape[1]):
            new_values[idx][i] = interpolators[i](point)
    
    return torch.tensor(new_values)

def unravel_tensor_final_v4(grid_num, data, pred, solution):
    is_3d = len(grid_num) == 3
    is_2d = len(grid_num) == 2
    if not (is_3d or is_2d):
        raise ValueError("Only 2D or 3D data is supported.")
    
    interpolated_pred = interpolate_corrected_v10(pred, grid_num)
    interpolated_solution = interpolate_corrected_v10(solution, grid_num)
    
    return interpolated_pred, interpolated_solution

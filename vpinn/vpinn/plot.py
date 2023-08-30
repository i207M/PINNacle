import torch
import re
import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib.tri import Triangulation
from .geom import rec, circle, cube, sphere, disk # NOTE: not all plotting styles of geoms are defined.
from .geomtime import timeline, timeplane

ONE_FIG_WIDTH = 7
ONE_FIG_HEIGHT = 6


def plot_2d(xx, yy, solution, prediction, res, channel, model_name, epoch):
    x = xx.detach().numpy() if torch.is_tensor(xx) else xx
    y = yy.detach().numpy() if torch.is_tensor(yy) else yy
    xx = torch.from_numpy(x).reshape(-1, 1)
    yy = torch.from_numpy(y).reshape(-1, 1)


    fig, axs = plt.subplots(nrows=channel, ncols=3, figsize=(3 * ONE_FIG_WIDTH, channel * ONE_FIG_HEIGHT))
    axes = axs.flatten()

    value = [solution.detach().numpy(), prediction.detach().numpy(), res.detach().numpy()]
    name = ['Solution', 'Prediction', 'Residual']
    
    for i in range(channel):
        for j in range(len(value)):
            image = axes[i * 3 + j].scatter(x, y, c=value[j][:, i], cmap='jet')
            axes[i * 3 + j].set_title(name[j])
            fig.colorbar(image, ax=axes[i * 3 + j])

    plt.savefig(f'fig/{model_name}, epoch={epoch}, error={torch.norm(solution - prediction) / torch.norm(solution) * 100:.2f}%.png')
    return [torch.cat([xx, yy], dim=1), 'plot_2d', prediction, solution]

def plot_3d_scatter(xx, yy, zz, solution, prediction, res, channel, model_name, epoch):
    x = xx.detach().numpy() if torch.is_tensor(xx) else xx
    y = yy.detach().numpy() if torch.is_tensor(yy) else yy
    z = zz.detach().numpy() if torch.is_tensor(zz) else zz

    fig, axs = plt.subplots(nrows=channel, ncols=3, subplot_kw={'projection': '3d'}, figsize=(3 * ONE_FIG_WIDTH, channel * ONE_FIG_HEIGHT))
    axes = axs.flatten()
    
    names = ['Solution', 'Prediction', 'Residual']
    values = [solution.detach().numpy(), prediction.detach().numpy(), res.detach().numpy()]
    
    for i in range(channel):
        for j in range(3):
            image = axes[i * 3 + j].scatter(x, y, z, c=values[j][:, i], cmap='jet')
            axes[i * 3 + j].set_title(names[j])
            fig.colorbar(image, ax=axes[i * 3 + j], shrink=0.5)

    plt.savefig(f'fig/{model_name}, epoch={epoch}, error={torch.norm(solution - prediction) / torch.norm(solution) * 100:.2f}%.png')
    return [torch.cat([xx, yy, zz], dim=1), 'plot_3d_scatter', prediction, solution]

def plot_3d_layer(grid_data, layers, model_name, channel, err, epoch):
    # note: The case for channel > 1 hasn't been implemetented.
    unique_x = torch.linspace(torch.min(grid_data[:, 0]), torch.max(grid_data[:, 0]), layers)
    unique_y = torch.linspace(torch.min(grid_data[:, 1]), torch.max(grid_data[:, 1]), layers)
    unique_z = torch.linspace(torch.min(grid_data[:, 2]), torch.max(grid_data[:, 2]), layers)
    
    fig, axs = plt.subplots(3, layers, figsize=(layers * ONE_FIG_WIDTH, 3 * ONE_FIG_HEIGHT))  # 3 rows for each plane type, 5 columns for each unique coordinate
    fig.suptitle(f'3D Layer Plots for {model_name}', fontsize=16)

    for i, x in enumerate(unique_x):
        mask = grid_data[:, 0] == x
        yz_plane = grid_data[mask][:, 1:]
        sc = axs[0, i].scatter(yz_plane[:, 0], yz_plane[:, 1], c=yz_plane[:, 2], cmap='jet')
        axs[0, i].set_title(f'YZ plane at x={x.item()}')
        fig.colorbar(sc, ax=axs[0, i])

    for i, y in enumerate(unique_y):
        mask = grid_data[:, 1] == y
        xz_plane = grid_data[mask][:, [0, 2, 3]]
        sc = axs[1, i].scatter(xz_plane[:, 0], xz_plane[:, 1], c=xz_plane[:, 2], cmap='jet')
        axs[1, i].set_title(f'XZ plane at y={y.item()}')
        fig.colorbar(sc, ax=axs[1, i])

    for i, z in enumerate(unique_z):
        mask = grid_data[:, 2] == z
        xy_plane = grid_data[mask][:, [0, 1, 3]]
        sc = axs[2, i].scatter(xy_plane[:, 0], xy_plane[:, 1], c=xy_plane[:, 2], cmap='jet')
        axs[2, i].set_title(f'XY plane at z={z.item()}')
        fig.colorbar(sc, ax=axs[2, i])

    plt.tight_layout()
    plt.savefig(f'fig/{model_name}_3d_layer_plot, epoch={epoch}, error={err * 100:.2f}%.png')
    plt.close(fig)

def plot_from_formula(net, geom, u, model_name, channel=1, N=200, layers=None, style='scatter', epoch=-1):
    if isinstance(geom, rec):
        net.cpu()
        x1, x2, y1, y2 = (geom.x1, geom.x2, geom.y1, geom.y2)
        x = torch.linspace(x1, x2, N)
        y = torch.linspace(y1, y2, N)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        xy = torch.cat([xx, yy], dim=1)
        solution = u(xy)
        prediction = net(xy)
        res = prediction - u(xy)

        return plot_2d(xx, yy, solution, prediction, res, channel, model_name, epoch)
    
    if isinstance(geom, timeline):
        net.cpu()
        x1, x2, t1, t2 = (geom.x1, geom.x2, 0, geom.t)
        x = torch.linspace(x1, x2, N)
        t = torch.linspace(t1, t2, N)
        xx, tt = torch.meshgrid(x, t, indexing='ij')
        xx = xx.reshape(-1, 1)
        tt = tt.reshape(-1, 1)
        xt = torch.cat([xx, tt], dim=1)
        solution = u(xt)
        prediction = net(xt)
        res = prediction - u(xt)

        return plot_2d(xx, tt, solution, prediction, res, channel, model_name, epoch)
    
    if isinstance(geom, timeplane):
        net = net.cpu()
        x = torch.linspace(geom.x1, geom.x2, N)
        y = torch.linspace(geom.y1, geom.y2, N)
        t = torch.tensor(layers, dtype=x.dtype)
        xx, yy, tt = torch.meshgrid(x, y, t, indexing='ij')
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        tt = tt.reshape(-1, 1)
        xyt = torch.cat([xx, yy, tt], dim=1)
        prediction = net(xyt)
        solution = u(xyt)
        err = torch.norm(solution - prediction) / torch.norm(solution)
        plot_time_layer(torch.cat([xyt, solution], dim=1), net, layers, model_name, err, channel, style, epoch)
        return [xyt, 'plot_timeplane_from_formula', prediction, solution]
        

def plot_time_layer(data, net, time_plane, model_name, error, channel, style='scatter', epoch=-1):
    fig, axs = plt.subplots(nrows=2 * channel, ncols=len(time_plane), figsize=(len(time_plane) * ONE_FIG_WIDTH, 2 * channel * ONE_FIG_HEIGHT))
    axes = axs.flatten()

    for i in range(len(time_plane)):
        mask = (data[:, 2] == time_plane[i])
        layer = data[mask]
        for j in range(channel):
            if channel == 1:
                variable_name = 'u'
            else:
                variable_name = f'u{j + 1}'
            
            solution_pos = i + j * 2 * len(time_plane)
            prediction_pos = i + (j * 2 + 1) * len(time_plane)
            
            if style == 'scatter':
                image1 = axes[solution_pos].scatter(layer[:,0].numpy(), layer[:,1].numpy(), c=layer[:,3 + j].numpy(), cmap='jet')
                axes[solution_pos].set_title(f'{variable_name} Solution, t={time_plane[i]}')
                fig.colorbar(image1, ax=axes[solution_pos])
                    
                image2 = axes[prediction_pos].scatter(layer[:,0].numpy(), layer[:,1].numpy().reshape(-1), c=(net(layer[:,0:3])[:,j]).detach().numpy(), cmap='jet')
                axes[prediction_pos].set_title(f'{variable_name} Prediction, t={time_plane[i]}')
                fig.colorbar(image2, ax=axes[prediction_pos])
            elif style == 'triangulation' or style == 'Triangulation':
                tri = Triangulation(layer[:,0].numpy(), layer[:,1].numpy())
                
                image1 = axes[solution_pos].tripcolor(tri, layer[:,3 + j].numpy(), cmap='jet', edgecolors='k')
                axes[solution_pos].set_title(f'{variable_name} Solution, t={time_plane[i]}')
                fig.colorbar(image1, ax=axes[solution_pos])
                    
                image2 = axes[prediction_pos].tripcolor(tri, (net(layer[:,0:3])[:,j]).detach().numpy().reshape(-1), cmap='jet', edgecolors='k')
                axes[prediction_pos].set_title(f'{variable_name} Prediction, t={time_plane[i]}')
                fig.colorbar(image2, ax=axes[prediction_pos])
                
            else:
                raise TypeError(f'\'{style}\' is not defined.')
        
    plt.savefig(f'fig/{model_name}, epoch={epoch}, error={error * 100:.2f}%.png')
    

def get_timestep(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        
    for line in lines:
        if line.startswith("% X") or line.startswith("% x"):
            matches = re.findall(r't=\d*\.?\d+', line)
            if len(matches) >= 4:  # at least 2 time steps are ned
                t1 = float(matches[0].split('=')[1])
                t2 = float(matches[1].split('=')[1])
                t3 = float(matches[2].split('=')[1])
                t4 = float(matches[3].split('=')[1])
                return t2 - t1 if t2 - t1 != 0 else (t3 - t1 if t3 - t1 != 0 else t4 - t1)  # return time step
    raise ValueError('Failed to find suitable time step')  # Exception will be throwed in case of no time steo found

def plot_from_data(net, geom, data_name, model_name, channel=1, grid_data=None, layers=None, style='scatter', epoch=-1, scale=1):
    net.cpu()
    data_name = 'ref/' + data_name
    data = np.loadtxt(data_name, skiprows=9)

    if isinstance(geom, timeline):
        if data.shape[1] > 10:
            # usually there are dozens of time line in this data format
            # get x、y、u of solution
            x_ = data[:, 0]
            x = []
            t = []
            u = []
            
            time_diff = get_timestep(data_name)
            for i in range(data.shape[1] - 1):
                x.append(copy.deepcopy(x_))
                t.append([i * time_diff for _ in range(len(x_))])
                u.append(data[:, i + 1])

            x = np.concatenate(x)
            t = np.concatenate(t)
            u = np.concatenate(u)
            
        else:
            # in this case, the time layers is given in a specific column
            x = data[:, 0]
            t = data[:, 1]
            u = []
            for i in range(channel):
                u.append(data[:, 2 + i].reshape(-1, 1))
            u = np.hstack(u)
            
        xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
        yy = torch.from_numpy(t).reshape(-1, 1).type(torch.float)
        uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)
        
        prediction = net(torch.cat([xx, yy], dim=1))
        solution = uu
        res = solution - prediction

        return plot_2d(xx, yy, solution, prediction, res, channel, model_name, epoch)
        
    
    if isinstance(geom, timeplane):
        
        if data.shape[1] > 10:
            # usually there are dozens of time plane in this data format
            # get x、y、u of solution
            x_ = data[:, 0]
            y_ = data[:, 1]
            x = []
            y = []
            t = []
            u = [[] for _ in range(int((data.shape[1] - 2) / channel))]
            
            time_diff = get_timestep(data_name)
            for i in range(int((data.shape[1] - 2) / channel)):
                x.append(copy.deepcopy(x_))
                y.append(copy.deepcopy(y_))
                t.append([i * time_diff for _ in range(len(x_))])
                
                for j in range(channel):
                    u[i].append(data[:, channel * i + 2 + j].reshape(-1, 1))
                u[i] = np.hstack(u[i])
                
            x = np.concatenate(x)
            y = np.concatenate(y)
            t = np.concatenate(t)
            u = np.concatenate(u)
        else:
            # in this case, the time layers is given in a specific column
            x = data[:, 0]
            y = data[:, 1]
            t = data[:, 2]
            u = []
            for i in range(channel):
                u.append(data[:, 3 + i].reshape(-1, 1))
            u = np.hstack(u)

        xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
        yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
        tt = torch.from_numpy(t).reshape(-1, 1).type(torch.float)
        uu = torch.from_numpy(u).reshape(-1, channel).type(torch.float)
        
        prediction = net(torch.cat([xx, yy, tt], dim=1))
        solution = uu
        res = solution - prediction
        
        if layers is None:
            raise TypeError('The default plot style of time PDE is \'layer\', but no layers are given.\nPlease provide \'layers\' argument.\n')
        
        plot_time_layer(torch.cat([xx, yy, tt, uu], dim=1), net, layers, model_name, torch.norm(solution - prediction) / torch.norm(solution), channel, style, epoch)

        return [torch.cat([xx, yy, tt], dim=1), 'plot_time_plane_from_data', prediction, solution]
    
    if isinstance(geom, rec):
        # get x、y、u of solution
        # scale the domain
        x = data[:, 0] * scale
        y = data[:, 1] * scale
        u = []
        for i in range(channel):
            u.append(data[:, 2 + i])

        u = np.stack(u).T
        
        xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
        yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
        uu = torch.from_numpy(u).reshape(-1, channel).type(torch.float)
        
        prediction = net(torch.cat([xx, yy], dim=1))
        solution = uu
        res = solution - prediction

        return plot_2d(x, y, solution, prediction, res, channel, model_name, epoch)
    
    if isinstance(geom, cube):
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        u = data[:, 3]
        
        xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
        yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
        zz = torch.from_numpy(z).reshape(-1, 1).type(torch.float)
        uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)
        
        prediction = net(torch.cat([xx, yy, zz], dim=1))
        solution = uu
        res = solution - prediction
        err = torch.norm(res) / torch.norm(solution)
        if grid_data == None:
            return plot_3d_scatter(xx, yy, zz, solution, prediction, res, channel, model_name)
        else:
            ret = [torch.cat([xx, yy, zz], dim=1), 'plot_3d_from_data', prediction, solution]
            prediction = net(grid_data)
            plot_3d_layer(torch.cat([grid_data, prediction.detach()], dim=1), layers, model_name, channel, err=err, epoch=epoch)
            return ret
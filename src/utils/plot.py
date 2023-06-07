import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
from src.pde.baseclass import BaseTimePDE

epsilon = 1e-10


def mean_squared_error_outlier(y_true, y_pred):
    '''
    MSE calculator.
    '''
    error = np.ravel((y_true - y_pred)**2)
    error = np.sort(error)[:-len(error) // 1000]
    return np.mean(error)


def plot_distribution_log(data, xlabel, ylabel, path, title=''):
    '''
    plot the distribution of data with log-scale.
    '''
    plt.cla()
    plt.figure()
    _, bins, _ = plt.hist(data, bins=30)
    plt.close()
    plt.cla()
    plt.figure()
    logbins = np.logspace(np.log10(bins[0] + epsilon), np.log10(bins[-1] + epsilon), len(bins))
    plt.hist(data, bins=logbins)
    plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axvline(x=np.mean(data), c='r', ls='--', lw=2)
    plt.title(title)
    plt.savefig(path)
    plt.close()


def plot_distribution(data, xlabel, ylabel, path, title=''):
    '''
    plot the distribution of data.
    '''
    plt.cla()
    plt.figure()
    plt.hist(data, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axvline(x=np.mean(data), c='r', ls='--', lw=2)
    plt.title(title)
    plt.savefig(path)
    plt.close()


def plot_points(data, xlabel, ylabel, path):
    '''
    Scatter points
    '''
    plt.cla()
    plt.figure()
    plt.axis('equal')
    x, y = zip(*list(data))
    plt.plot(x, y, 'bo')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()


def plot_lines(data, path, xlabel="", ylabel="", labels=None, xlog=False, ylog=False, title='', sort_=False):
    '''
    Lines
    '''
    plt.cla()
    plt.figure()
    if labels is None:
        labels = ["" for _ in range(len(data) - 1)]
    for i in range(1, len(data)):
        if sort_:
            x = np.array(data[0])
            y = np.array(data[i])
            sorted_indices = np.argsort(x)
            sorted_x = x[sorted_indices]
            sorted_y = y[sorted_indices]
            plt.plot(sorted_x, sorted_y, label=labels[i - 1])
        else:
            plt.plot(data[0], data[i], label=labels[i - 1])
    plt.legend()
    if ylog: plt.yscale('log')
    if xlog: plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path)
    plt.close()


def plot_heatmap(x, y, z, path=None, vmin=None, vmax=None, num=100, title='', xlabel='x', ylabel='y', show=False, pde=None):
    '''
    Plot heat map for a 3-dimension data
    '''
    plt.cla()
    plt.figure()
    xx = np.linspace(np.min(x), np.max(x), num)
    yy = np.linspace(np.min(y), np.max(y), num)
    xx, yy = np.meshgrid(xx, yy)

    vals = interpolate.griddata(np.array([x, y]).T, np.array(z), (xx, yy), method='cubic')
    vals_0 = interpolate.griddata(np.array([x, y]).T, np.array(z), (xx, yy), method='nearest')
    vals[np.isnan(vals)] = vals_0[np.isnan(vals)]
    if pde is not None:
        if isinstance(pde, BaseTimePDE):
            # vals[~pde.geomtime.inside(np.stack((xx, yy), axis=2))] = np.nan
            assert pde.geomtime.dim == 2
        else:  # for BasePDE
            vals[~pde.geom.inside(np.stack((xx, yy), axis=2))] = np.nan

    vals = vals[::-1, :]  # reverse y coordinate: for imshow, (0,0) show at top left.

    fig = plt.imshow(vals, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], aspect='auto', interpolation='bicubic', vmin=vmin, vmax=vmax)
    fig.axes.set_autoscale_on(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    if path:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def plot_3dheatmap(x, y, z, values, path, time_split=6, title='', xlabel='x', ylabel='y', zlabel='z', no_interpolate=False):
    bbox = [x.min(), x.max(), y.min(), y.max(), z.min(), z.max()]
    xx, yy, zz = np.linspace(bbox[0], bbox[1]), np.linspace(bbox[2], bbox[3]), np.linspace(bbox[4], bbox[5], time_split)
    xx, yy, zz = np.meshgrid(xx, yy, zz)
    vals = interpolate.griddata(np.array([x, y, z]).T, values, (xx, yy, zz), method='nearest')  # NOTE: for speed
    vals_0 = interpolate.griddata(np.array([x, y, z]).T, values, (xx, yy, zz), method='nearest')
    vals[np.isnan(vals)] = vals_0[np.isnan(vals)]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=10, azim=45)
    norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())
    for i in range(time_split):
        ax.contourf(xx[:, :, i], yy[:, :, i], vals[:, :, i], zdir='z', offset=zz[0][0][i], cmap="coolwarm", norm=norm)

    ax.set(xlim=(bbox[0], bbox[1]), ylim=(bbox[2], bbox[3]), zlim=(bbox[4], bbox[5]), xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax)
    plt.title(title)
    plt.savefig(path)
    plt.close()


# TODO: advanced options
def plot_streamline(x, y, u, v, path=None, vmin=None, vmax=None, title='', xlabel='x', ylabel='y', show=False):
    '''
    Plot heat map for a 3-dimension data
    '''
    plt.cla()
    plt.figure()
    xx = np.linspace(np.min(x), np.max(x))
    yy = np.linspace(np.min(y), np.max(y))
    xx, yy = np.meshgrid(xx, yy)

    us = interpolate.griddata(np.array([x, y]).T, np.array(u), (xx, yy), method='cubic')
    us_0 = interpolate.griddata(np.array([x, y]).T, np.array(u), (xx, yy), method='nearest')
    us[np.isnan(us)] = us_0[np.isnan(us)]

    vs = interpolate.griddata(np.array([x, y]).T, np.array(v), (xx, yy), method='cubic')
    vs_0 = interpolate.griddata(np.array([x, y]).T, np.array(v), (xx, yy), method='nearest')
    vs[np.isnan(vs)] = vs_0[np.isnan(vs)]

    fig = plt.streamplot(xx, yy, us, vs)
    # fig.axes.set_autoscale_on(False)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    if path:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def plot_state(pde, train_state, output_dir, is_best=False, fast=False):
    """Plot the current/best result of the smallest training loss.
    """
    if isinstance(train_state.X_train, (list, tuple)):
        raise NotImplementedError("Error: The network has multiple inputs, and plotting such result han't been implemented.")

    def merge_values(values):
        if values is None:
            return None
        return np.hstack(values) if isinstance(values, (list, tuple)) else values

    x_dim = pde.input_dim
    y_dim = pde.output_dim
    x_test = train_state.X_test
    y_test = merge_values(train_state.best_y) if is_best else merge_values(train_state.y_pred_test)
    if x_dim == 1:  # maintain increasing order for plotting
        idx = np.argsort(train_state.X_test[:, 0])
        x_test = x_test[idx, :]
        y_test = y_test[idx, :]
    if pde.ref_sol is not None:
        x_data = x_test
        y_data = pde.ref_sol(x_test)
    elif pde.ref_data is not None:
        x_data = pde.ref_data[:, :pde.input_dim]
        y_data = pde.ref_data[:, pde.input_dim:]
    else:
        x_data = y_data = None
    
    np.savetxt(output_dir + "model_output.txt", np.concatenate((x_test, y_test), axis=1), \
               header=f"pde: {type(pde).__name__}, x_dim: {x_dim}, y_dim: {y_dim}") # save model output when plotting

    # Regression plot
    # 1D
    if x_dim == 1:
        for i in range(y_dim):
            plt.figure()
            plt.plot(x_test[:, 0], y_test[:, i], "--r", label=f"Prediction_{pde.output_config[i]['name']}")
            if y_data is not None:
                plt.plot(x_data[:, 0], y_data[:, i], "-k", label=f"True_{pde.output_config[i]['name']}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.savefig(output_dir + pde.output_config[i]['name'] + '_pred.png')
            plt.close()

            if y_data is not None:
                plt.figure()
                if pde.ref_sol is not None:
                    y = y_test[:, i]
                else:
                    y = interpolate.griddata(x_test, y_test[:, i], x_data, method='cubic')
                    y0 = interpolate.griddata(x_test, y_test[:, i], x_data, method='nearest')
                    y[np.isnan(y)] = y0[np.isnan(y)]
                plt.plot(x_data[:, 0], y - y_data[:, i], "r", label=f"Error_{pde.output_config[i]['name']}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.savefig(output_dir + pde.output_config[i]['name'] + '_err.png')
                plt.close()
    # 2D
    elif x_dim == 2:
        for i in range(y_dim):
            save_path = output_dir + pde.output_config[i]['name'] + '_pred.png'
            error_save_path = output_dir + pde.output_config[i]['name'] + '_err.png'
            plot_heatmap(x_test[:, 0], x_test[:, 1], y_test[:, i], save_path, title=f"Prediction for {pde.output_config[i]['name']}", pde=pde)
            if x_data is not None:
                if pde.ref_sol is not None:
                    plot_heatmap(
                        x_test[:, 0],
                        x_test[:, 1],
                        y_test[:, i] - y_data[:, i],
                        error_save_path,
                        title=f"Error for {pde.output_config[i]['name']}",
                        pde=pde
                    )
                else:
                    y = interpolate.griddata(x_test, y_test[:, i], x_data, method='cubic')
                    y0 = interpolate.griddata(x_test, y_test[:, i], x_data, method='nearest')
                    y[np.isnan(y)] = y0[np.isnan(y)]
                    plot_heatmap(
                        x_data[:, 0], x_data[:, 1], y - y_data[:, i], error_save_path, title=f"Error for {pde.output_config[i]['name']}", pde=pde
                    )
    # 3D
    elif x_dim == 3:
        for i in range(y_dim):
            save_path = output_dir + pde.output_config[i]['name'] + '_pred.png'
            error_save_path = output_dir + pde.output_config[i]['name'] + '_err.png'
            plot_3dheatmap(x_test[:, 0], x_test[:, 1], x_test[:, 2], y_test[:, i], save_path, title=f"Prediction for {pde.output_config[i]['name']}")
            if x_data is not None:
                if pde.ref_sol is not None:
                    plot_3dheatmap(
                        x_test[:, 0],
                        x_test[:, 1],
                        x_test[:, 2],
                        y_test[:, i] - y_data[:, i],
                        error_save_path,
                        title=f"Error for {pde.output_config[i]['name']}"
                    )
                else:
                    if not fast:
                        y = interpolate.griddata(x_test, y_test[:, i], x_data)
                        y0 = interpolate.griddata(x_test, y_test[:, i], x_data, method='nearest')
                        y[np.isnan(y)] = y0[np.isnan(y)]
                        plot_3dheatmap(
                            x_data[:, 0],
                            x_data[:, 1],
                            x_data[:, 2],
                            y - y_data[:, i],
                            error_save_path,
                            title=f"Error for {pde.output_config[i]['name']}"
                        )
                    else:  # using training points (if data is **dense**)
                        y = interpolate.griddata(x_data, y_data[:, i], x_test, method='nearest')
                        plot_3dheatmap(
                            x_test[:, 0],
                            x_test[:, 1],
                            x_test[:, 2],
                            y_test[:, i] - y,
                            error_save_path,
                            title=f"Error for {pde.output_config[i]['name']}"
                        )


def plot_loss_history(pde, loss_history, output_dir, loss_weights=None):
    """Plot the training and testing loss history.

    Note:
        You need to call ``plt.show()`` to show the figure.

    Args:
        loss_history: ``LossHistory`` instance. The first variable returned from
            ``Model.train()``.
        fname (string): If `fname` is a string (e.g., 'loss_history.png'), then save the
            figure to the file of the file name `fname`.
    """
    if loss_weights is not None:
        loss_train_sum = np.sum(loss_history.loss_train / loss_weights, axis=1)
        loss_test = loss_history.loss_test / loss_weights
        loss_test_sum = np.sum(loss_test, axis=1)
        loss_type = "unweighted_loss"
    else:
        loss_train_sum = np.sum(loss_history.loss_train, axis=1)
        loss_test_sum = np.sum(loss_history.loss_test, axis=1)
        loss_test = np.array(loss_history.loss_test)
        loss_type = "loss"

    plt.figure()
    plt.semilogy(loss_history.steps, loss_train_sum, label="Train loss")
    plt.semilogy(loss_history.steps, loss_test_sum, label="Test loss")
    for i in range(len(loss_history.metrics_test[0])):
        plt.semilogy(
            loss_history.steps,
            np.array(loss_history.metrics_test)[:, i],
            label="Test metric",
        )
    plt.xlabel("# Steps")
    plt.legend()
    plt.savefig(output_dir + f"{loss_type}.png")
    plt.close()

    plt.figure()
    for i in range(pde.num_loss):
        plt.semilogy(loss_history.steps, loss_test[:, i], label=pde.loss_config[i]['name'])

    plt.xlabel("# Steps")
    plt.legend()
    plt.savefig(output_dir + f"{loss_type}_detail.png")
    plt.close()

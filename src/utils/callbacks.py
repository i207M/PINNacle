import logging
import os
import torch
import numpy as np

from deepxde.callbacks import Callback
from deepxde.utils.internal import list_to_str
from src.utils import plot

logger = logging.getLogger(__name__)


class PlotCallback(Callback):

    def __init__(self, log_every=None, verbose=False, fast=False):
        super(PlotCallback, self).__init__()

        self.log_every = log_every
        self.verbose = verbose
        self.fast = fast
        self.valid_epoch = 0

    def plot(self, save_path):
        train_state = self.model.train_state
        plot.plot_state(self.model.pde, train_state, save_path, is_best=False, fast=self.fast)

    def on_train_begin(self):
        self.base_save_path = self.model.model_save_path + "/"
        if not os.path.exists(self.base_save_path):
            os.mkdir(self.base_save_path)

    def on_epoch_end(self):
        self.valid_epoch += 1
        if self.log_every is None or self.valid_epoch % self.log_every != 0:
            return
        if self.verbose:
            print("Plotting at epoch {} ...".format(self.valid_epoch))

        save_path = self.base_save_path + str(self.valid_epoch) + '/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.plot(save_path)

    def on_train_end(self):
        if self.verbose:
            print("Plotting at train end ...")
        self.plot(self.base_save_path)


class LossCallback(Callback):

    def __init__(self, use_weighted_loss=False, verbose=False):
        super(LossCallback, self).__init__()
        self.log_every = None
        self.weighted = use_weighted_loss
        self.verbose = verbose
        self.valid_epoch = 0
        self.loss_weights = []

    def on_train_begin(self):
        self.log_every = self.model.display_every
        if not self.weighted:
            if self.model.losshistory.loss_weights is not None:
                self.loss_weights.append(self.model.losshistory.loss_weights)
            else:
                self.loss_weights.append(np.ones(self.model.pde.num_loss))

    def on_epoch_end(self):
        self.valid_epoch += 1
        if self.weighted or self.valid_epoch % self.log_every != 0:  # if use weighted loss, no need to record weights
            return

        if self.model.losshistory.loss_weights is not None:
            self.loss_weights.append(self.model.losshistory.loss_weights.copy())
        else:
            self.loss_weights.append(np.ones(self.model.pde.num_loss))

        if self.verbose:
            loss_weight = self.loss_weights[-1]
            loss_train = self.model.losshistory.loss_train[-1] / loss_weight
            loss_test = self.model.losshistory.loss_test[-1] / loss_weight
            print('Unweighted Loss: {}  {} Weights: {}'.format(
                list_to_str(loss_train),
                list_to_str(loss_test),
                list_to_str(loss_weight),
            ))

    def on_train_end(self):
        save_path = self.model.model_save_path + "/"
        loss_history = self.model.losshistory
        if self.weighted:
            loss = np.hstack((
                np.array(loss_history.steps)[:, None],
                np.array(loss_history.loss_train),
                np.array(loss_history.loss_test),
            ))
            np.savetxt(save_path + "loss.txt", loss, header="step, loss_train, loss_test")
            plot.plot_loss_history(self.model.pde, loss_history, save_path)
        else:
            loss_weights = np.array(self.loss_weights)
            loss = np.hstack((
                np.array(loss_history.steps)[:, None],
                np.array(loss_history.loss_train) / loss_weights,
                np.array(loss_history.loss_test) / loss_weights,
                loss_weights,
            ))
            np.savetxt(save_path + "loss.txt", loss, header="step, loss_train, loss_test, loss_weight")
            plot.plot_loss_history(self.model.pde, loss_history, save_path)
            plot.plot_loss_history(self.model.pde, loss_history, save_path, loss_weights=loss_weights)


class TesterCallback(Callback):

    def __init__(self, log_every=100, verbose=True):
        super(TesterCallback, self).__init__()

        self.log_every = log_every
        self.verbose = verbose

        self.indexes = []
        self.maes = []
        self.mses = []
        self.l1res = []
        self.l2res = []

        self.epochs_since_last_resample = 0
        self.valid_epoch = 0
        self.disable = False

    def on_train_begin(self):
        self.save_path = self.model.model_save_path + "/"
        pde = self.model.pde
        if pde.ref_sol is not None:
            if pde.input_dim == 2:
                sample_points = 50
            elif pde.input_dim == 3:
                sample_points = 30
            else:
                sample_points = int(1e4**(1 / pde.input_dim))
            xlist = [np.linspace(pde.bbox[i * 2], pde.bbox[i * 2 + 1], sample_points) for i in range(pde.input_dim)]
            self.test_x = np.stack(np.meshgrid(*xlist), axis=-1).reshape(-1, pde.input_dim)
            self.test_y = pde.ref_sol(self.test_x)
        elif pde.ref_data is not None:
            nan_mask = np.isnan(pde.ref_data).any(axis=1)
            self.test_x = pde.ref_data[~nan_mask, :pde.input_dim]
            self.test_y = pde.ref_data[~nan_mask, pde.input_dim:]
        else:
            self.disable = True
            logger.info("No reference solution or data provided, skipping TesterCallback")
            return

        self.solution_l1 = np.abs(self.test_y).mean()
        self.solution_l2 = np.sqrt((self.test_y**2).mean())

    def on_epoch_end(self):
        self.epochs_since_last_resample += 1
        self.valid_epoch += 1
        if self.disable or self.log_every is None or self.epochs_since_last_resample < self.log_every:
            return
        self.epochs_since_last_resample = 0

        with torch.no_grad():
            y = self.model.predict(self.test_x)

        mse = ((y - self.test_y)**2).mean()
        mae = np.abs(y - self.test_y).mean()
        l1re = mae / self.solution_l1
        l2re = np.sqrt(mse) / self.solution_l2

        self.indexes.append(self.valid_epoch)
        self.mses.append(mse)
        self.maes.append(mae)
        self.l1res.append(l1re)
        self.l2res.append(l2re)

        if self.verbose:
            print('Validation: epoch {} MSE {:.5f} MAE {:.5f} L1RE {:.5f} L2RE {:.5f}'.format(self.valid_epoch, mse, mae, l1re, l2re))

    def on_train_end(self):
        if self.disable:
            return

        np.savetxt(
            self.save_path + 'errors.txt',
            np.array([self.indexes, self.maes, self.mses, self.l1res, self.l2res]).T,
            header="epochs, maes, mses, l1res, l2res"
        )

        plot.plot_lines([self.indexes, self.maes], xlabel="epochs", labels=['maes'], path=self.save_path + "maes.png", title="mean average error")
        plot.plot_lines([self.indexes, self.mses], xlabel="epochs", labels=['mses'], path=self.save_path + "mses.png", title="mean square error")
        plot.plot_lines([self.indexes, self.l1res, self.l2res],
                        xlabel="epochs",
                        labels=['l1re', 'l2re'],
                        path=self.save_path + "relerr.png",
                        title="relative error")

        self.indexes = []
        self.mses = []
        self.maes = []
        self.l1res = []
        self.l2res = []

        self.epochs_since_last_resample = 0
        self.valid_epoch = 0

#!/usr/bin/env python
# -*- coding:utf-8 _*-
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import fft
import random
import os
import torch

pi = np.pi


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_grid2d(x_range, y_range, N_x=50, N_y=50, grid_on=False):
    x = np.linspace(x_range[0], x_range[1], N_x)
    y = np.linspace(y_range[0], y_range[1], N_y)

    xx_, yy_ = np.meshgrid(x, y)

    xx, yy = xx_.reshape(-1), yy_.reshape(-1)
    xy = np.array([xx, yy]).T
    if grid_on:
        return xy, xx_, yy_
    else:
        return xy


def generate_burgers_2d_source_functions(k=1):
    K = 4
    L = 4
    N_res = 512

    def generate_initial_uv():
        x = np.linspace(0, L, N_res)
        xx, yy = np.meshgrid(x, x)
        u = 0
        A = np.random.randn(K, K)
        B = np.random.randn(K, K)

        for i in range(K):
            for j in range(K):
                u += A[i, j] * np.sin(2 * pi * (i * xx + j * yy)) + B[i, j] * np.cos(2 * pi * (i * xx + j * yy))
        return u, xx, yy

    u_, xx, yy = generate_initial_uv()
    v_, xx, yy = generate_initial_uv()
    uv_ = np.stack([u_, v_])
    c = np.random.randn(2)
    u = 2 * u_ / np.max(uv_) + c[0]
    v = 2 * v_ / np.max(uv_) + c[1]

    data_u, data_v, data_xx, data_yy = u.reshape(-1), v.reshape(-1), xx.reshape(-1), yy.reshape(-1)

    data_u = np.stack([data_xx, data_yy, data_u]).T
    data_v = np.stack([data_xx, data_yy, data_v]).T

    np.savetxt("./burgers2d_init_u_{}.txt".format(k), data_u)
    np.savetxt("./burgers2d_init_v_{}.txt".format(k), data_v)

    plt.figure()
    plt.imshow(u)
    plt.imshow(v)
    plt.colorbar()
    plt.show()


def fftind(size):
    """Returns a numpy array of shifted Fourier coordinates k_x k_y.

    Input args:
        size (integer): The size of the coordinate array to create
    Returns:
        k_ind, numpy array of shape (2, size, size) with:
            k_ind[0,:,:]:  k_x components
            k_ind[1,:,:]:  k_y components

    Example:

        print(fftind(5))

        [[[ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]]
        [[ 0  0  0  0  0]
        [ 1  1  1  1  1]
        [-3 -3 -3 -3 -3]
        [-2 -2 -2 -2 -2]
        [-1 -1 -1 -1 -1]]]

    """
    k_ind = np.mgrid[:size, :size] - int((size + 1) / 2)
    k_ind = scipy.fft.fftshift(k_ind)
    return k_ind


def gaussian_random_field(alpha=3.0, size=128, flag_normalize=True):
    """Returns a numpy array of shifted Fourier coordinates k_x k_y.

    Input args:
        alpha (double, default = 3.0):
            The power of the power-law momentum distribution
        size (integer, default = 128):
            The size of the square output Gaussian Random Fields
        flag_normalize (boolean, default = True):
            Normalizes the Gaussian Field:
                - to have an average of 0.0
                - to have a standard deviation of 1.0
    Returns:
        gfield (numpy array of shape (size, size)):
            The random gaussian random field

    Example:
    import matplotlib
    import matplotlib.pyplot as plt
    example = gaussian_random_field()
    plt.imshow(example)
    """

    # Defines momentum indices
    k_idx = fftind(size)

    # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power(k_idx[0] ** 2 + k_idx[1] ** 2 + 1e-10, -alpha / 4.0)
    amplitude[0, 0] = 0

    # Draws a complex gaussian random noise with normal
    # (circular) distribution
    noise = np.random.normal(size=(size, size)) + 1j * np.random.normal(size=(size, size))

    # To real space
    gfield = np.fft.ifft2(noise * amplitude).real

    # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield / np.std(gfield)

    return gfield


## seems something wrong, code by FNO data generation
# def GRF(alpha, tau, s):
#     xi = np.random.randn(s,s)
#
#     [K1, K2] = np.meshgrid(np.arange(s), np.arange(s))
#     coef = tau **(alpha - 1)* (pi ** 2 * (K1** 2 + K2** 2) + tau ** 2)** (-alpha / 2)
#     L = s * coef *xi
#     L[0,0] = 0
#
#     U = fftpack.idct(L,type=2)
#     return U


def generate_darcy_2d_coef():
    N_res = 256
    alpha = 4
    # tau = 8
    L = 1
    # norm_a = GRF(alpha, tau, N_res)
    norm_a = gaussian_random_field(alpha=alpha, size=N_res)
    # phinorm_a = np.exp(norm_a)
    phinorm_a = np.where(norm_a > 0, 12, 1)

    x = np.linspace(0, L, N_res)

    xx, yy = np.meshgrid(x, x)
    darcy_2d_data, xx, yy = phinorm_a.reshape(-1), xx.reshape(-1), yy.reshape(-1)
    darcy_2d_data = np.stack([xx, yy, darcy_2d_data]).T

    np.savetxt("./darcy_2d_coef_256.txt", darcy_2d_data)
    plt.figure()
    plt.imshow(phinorm_a)
    plt.colorbar()
    plt.imsave("./darcy_2d_coef_256.png", phinorm_a)
    plt.show()


def generate_heat_2d_coef():
    N_res = 256
    alpha = 4
    # tau = 8
    L = 1
    # norm_a = GRF(alpha, tau, N_res)
    norm_a = gaussian_random_field(alpha=alpha, size=N_res)
    phinorm_a = np.exp(norm_a)
    # phinorm_a = np.where(norm_a>0,12,1)

    x = np.linspace(0, L, N_res)

    xx, yy = np.meshgrid(x, x)
    darcy_2d_data, xx, yy = phinorm_a.reshape(-1), xx.reshape(-1), yy.reshape(-1)
    darcy_2d_data = np.stack([xx, yy, darcy_2d_data]).T

    np.savetxt("./heat_2d_coef_256.txt", darcy_2d_data)
    plt.figure()
    plt.imshow(phinorm_a)
    plt.colorbar()
    plt.imsave("./heat_2d_coef_256.png", phinorm_a)
    plt.show()


if __name__ == "__main__":
    for i in range(5):
        seed_everything(20220722 + i)
        generate_burgers_2d_source_functions(i)
    # generate_darcy_2d_coef()
    # generate_heat_2d_coef()

from scipy import fftpack
import numpy as np


def generate_possion_a_coef(shape=(10, 10), rang=(-2, 3)):
    a_coef = np.random.random(shape) * (rang[1] - rang[0]) + rang[0]
    return np.exp(a_coef * np.log(10))  # exponential distribution among [10**-2, 10**3]


def fftind(size):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.

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
    k_ind = fftpack.fftshift(k_ind)
    return (k_ind)


def gaussian_random_field(alpha=3.0, size=128, flag_normalize=True):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.

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
    amplitude = np.power(k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha / 4.0)
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


def generate_darcy_2d_coef(N_res, alpha, bbox):
    norm_a = gaussian_random_field(alpha=alpha, size=N_res)
    phinorm_a = np.where(norm_a > 0, 12, 1)

    x = np.linspace(bbox[0], bbox[1], N_res)
    y = np.linspace(bbox[2], bbox[3], N_res)

    xx, yy = np.meshgrid(x, y)
    darcy_2d_data, xx, yy = phinorm_a.reshape(-1), xx.reshape(-1), yy.reshape(-1)
    darcy_2d_data = np.stack([xx, yy, darcy_2d_data]).T

    return darcy_2d_data


def generate_heat_2d_coef(N_res, alpha, bbox):
    norm_a = gaussian_random_field(alpha=alpha, size=N_res)
    phinorm_a = np.exp(norm_a)

    x = np.linspace(bbox[0], bbox[1], N_res)
    y = np.linspace(bbox[2], bbox[3], N_res)

    xx, yy = np.meshgrid(x, y)
    darcy_2d_data, xx, yy = phinorm_a.reshape(-1), xx.reshape(-1), yy.reshape(-1)
    darcy_2d_data = np.stack([xx, yy, darcy_2d_data]).T

    return darcy_2d_data


if __name__ == "__main__":
    data = generate_possion_a_coef()
    np.savetxt("ref/poisson_a_coef.dat", data)

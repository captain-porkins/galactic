from scipy.constants import mu_0, epsilon_0, pi
from numba import jit
import numpy as np


class Dipole(object):
    def __init__(self, moment_vector=None, moment_magnitude=None, moment_orientation=None, charge=None, mass=None):
        if moment_vector is not None:
            if moment_orientation is not None:
                raise ValueError('Do not specify both a moment vector and a moment orientation')
            else:
                self.moment_vector = moment_vector
        elif moment_magnitude is not None:
            moment_orientation = moment_orientation or (0, 0, 1)
            self.moment_vector = np.multiply(moment_magnitude, moment_orientation)

        if charge is not None:
            raise NotImplementedError('Charged dipoles not yet implemented')

        if mass is not None:
            raise NotImplementedError('Massive dipoles not yet implemented')

        self.time_varying = False

    @jit(nopython=True, cache=True)
    def magnetic_field(self, position):
        scalar_position = np.linalg.norm(position)
        prefactor = (mu_0 / (4 * pi * scalar_position ** 3))

        first_term = np.multiply(3 * np.dot(self.moment_vector, position) / (scalar_position**2), position)
        return prefactor * np.subtract(first_term, self.moment_vector)

    @jit(nopython=True, cache=True)
    def electric_field(self, position):
        return np.array((0, 0, 0))


if __name__ == '__main__':
    dipole = Dipole(moment_magnitude=10)
    b = dipole.magnetic_field((1, 1, 1))
from numba import jit
import numpy as np
from galactic.fields import Dipole
from scipy.constants import c, proton_mass, elementary_charge


FILE_OUTPUT = r'C:\Users\Povey\Documents\test_galactic.csv'


class CartesianVector(object):
    def __init__(self, sequence):
        if len(sequence) == 3:
            self._sequence = sequence
        else:
            raise ValueError

    @property
    def x(self):
        return self._sequence[0]

    @property
    def y(self):
        return self._sequence[1]

    @property
    def z(self):
        return self._sequence[2]


class Particle(object):
    def __init__(self, start_position, start_velocity, charge, mass):
        self.position1 = np.array(start_position, dtype=float)
        self.position2 = np.array(start_position, dtype=float)

        self.velocity1 = np.array(start_velocity, dtype=float)
        self.velocity2 = np.array(start_velocity, dtype=float)

        self.charge = charge
        self.mass = mass


def galactic_iterator(run_time, dt):
    time = 0  # seconds

    while time <= run_time:
        yield time
        time += dt


def _get_matrix(prefactor, b):
    b_x, b_y, b_z = b
    matrix = np.matrix([(1, -prefactor * b_z, prefactor * b_y),
                        (prefactor * b_z, 1, -prefactor * b_x),
                        (-prefactor * b_y, prefactor * b_x, 1)])

    return matrix


def _calc_velocity(current_velocity, prefactor, electric_field, magnetic_field):
    matrix = _get_matrix(prefactor, magnetic_field)
    inv_matrix = matrix ** -1
    vel = np.matmul(np.matmul(inv_matrix, matrix.T), current_velocity) + 2 * prefactor * np.matmul(inv_matrix, electric_field)

    vel = np.array(vel)

    return vel[0]


def galactic(particles, field_source, run_time, dt, iterations=3):
    with open(FILE_OUTPUT, 'w') as f:
        for time in galactic_iterator(run_time, dt):
            for particle in particles:
                f.write(', '.join(str(p) for p in particle.position1) + '\n')
                particle.position1 = particle.position2
                particle.velocity1 = particle.velocity2
                for iteration in range(iterations):
                    mid = 0.5 * (particle.position1 + particle.position2)

                    kwargs = {'position': mid}
                    if field_source.time_varying:
                        kwargs.update({'time': time})

                    e_mid = field_source.electric_field(**kwargs)
                    b_mid = field_source.magnetic_field(**kwargs)

                    prefactor = 0.5 * dt * particle.charge / particle.mass
                    particle.velocity2 = _calc_velocity(particle.velocity1, prefactor, e_mid, b_mid)
                    particle.position2 = particle.position1 + 0.5 * (particle.velocity1 + particle.velocity2) * dt


def numba_galactic(particles, field_source, run_time, dt, iterations=3):

    numba_particles = [[p.position1, p.position2, p.velocity1, p.veclocty2] for p in particles]

    @jit(nopython=True, cache=True)
    def _numba_galactic(_particles, _run_time):
        for time in galactic_iterator(_run_time, dt):
            for particle in _particles:
                particle[0] = particle[1]  # pos1 = pos2
                particle[2] = particle[3]  # vel1 = vel2
                for iteration in range(iterations):
                    pos1, pos2 = particle[0], particle[1]
                    vel1, vel2 = particle[2], particle[3]

                    mid = 0.5 * (pos1 + pos2)

                    kwargs = {'position': mid}
                    if field_source.time_varying:
                        kwargs.update({'time': time})

                    e_mid = field_source.electric_field(**kwargs)
                    b_mid = field_source.magnetic_field(**kwargs)

                    prefactor = 0.5 * dt * particle.charge / particle.mass
                    particle.velocity2 = _calc_velocity(particle.velocity1, prefactor, e_mid, b_mid)
                    particle.position2 = particle.position1 + 0.5 * (particle.velocity1 + particle.velocity2) * dt


        return particles


if __name__ == '__main__':
    dip = Dipole(moment_magnitude=1E8)
    rt2 = 2 ** 0.5
    rt3 = 3 ** 0.5
    particles = [Particle(start_position=(0.5, 0, 0),
                          start_velocity=(rt2 * 0.8 * c / rt3, 0, 0.8 * c / rt3),
                          charge=elementary_charge, mass=proton_mass)]
    galactic(particles=particles, field_source=dip, dt=1E-10, run_time=1E-5)
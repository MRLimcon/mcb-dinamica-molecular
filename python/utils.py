import numpy as np
from scipy.stats import maxwell
from numba import njit, jit, prange
from itertools import product

# Constants
e_0 = 0.05
k_b = 1.
g = -1.
alpha = 1e-6

# Aliases for logical operations
np_or = np.logical_or
np_and = np.logical_and


# Configuration class
class config_class:
    L = [0, 0, 0]
    T = 273.15
    l_box = 1.0
    delta_t = 0
    boxes = []
    area = 0

    def make_boxes(self, new_l: list[float]):
        # Calculate the amount of boxes
        amount = np.array(
            [new_l[0] // self.l_box, new_l[1] // self.l_box, new_l[2] // self.l_box]
        ).astype(int) + 1

        # Create an array of lists
        self.boxes = np.empty((amount[0], amount[1], amount[2]), dtype=object)

        # Initialize each element as an empty list
        for j in range(amount[0]):
            for k in range(amount[1]):
                for l in range(amount[2]):
                    self.boxes[j, k, l] = []

    def __init__(
        self, L: list[float], T: float = 273.15, sigma: float = 1, delta_t: float = 0.05
    ):
        self.L = L
        self.T = T
        self.l_box = 2 * sigma
        self.delta_t = delta_t

        # Calculate the area of the simulation space
        self.area = 2 * L[0] * L[1] + 2 * L[0] + L[2] + 2 * L[1] * L[2]

        # Initialize the boxes
        self.make_boxes(L)


# Particle class
class particles_class:
    v = []
    positions = []
    dims = []
    force = []
    m = 1
    A = 1
    B = 1
    scale = 0
    sigma = 0

    def make_scale(self, T):
        # Calculate the scaling factor for particle velocity
        self.scale = np.sqrt(T * k_b / self.m)

    def __init__(
        self,
        N: int,
        L: list[float],
        mass: float = 1.164026113e-21,
        sigma: float = 1,
        T: float = 273.15,
    ):
        self.m = mass
        self.sigma = sigma
        self.B = 6 * 4 * e_0 * (sigma**6)
        self.A = 12 * 4 * e_0 * (sigma**12)
        self.scale = np.sqrt(T * k_b / self.m)

        dims = [N, 3]
        self.dims = dims

        # Initialize particle positions randomly within the simulation space
        random = np.random.rand(N, 3)
        self.force = np.zeros(dims)
        self.positions = np.zeros(dims)
        self.positions[:, 0] = random[:, 0] * L[0]
        self.positions[:, 1] = random[:, 1] * L[1]
        self.positions[:, 2] = random[:, 2] * L[2]

        self.v = np.zeros(dims)

        # Generate initial particle velocities based on Maxwell distribution
        theta = np.random.rand(N) * 3.1415 * 2.0
        phi = np.random.rand(N) * 3.1415
        avg_v = np.sqrt(8 * k_b * T / (np.pi * self.m))

        self.v[:, 0] = np.cos(theta) * np.sin(phi) * avg_v
        self.v[:, 1] = np.sin(theta) * np.sin(phi) * avg_v
        self.v[:, 2] = np.cos(phi) * avg_v


# @njit(parallel=True, fastmath=True, error_model="numpy")
def flatten(nearby_boxes, i_s, j_s, k_s):
    val_list = []
    for i, j, k in product(i_s, j_s, k_s):
        val_list.extend(nearby_boxes[i, j, k])

    return np.array(val_list)


@njit(parallel=True, fastmath=True, error_model="numpy")
def calc_lennard_jones_force(particle, inside_positions, A, sigma):
    particle_force = np.zeros(3, dtype=np.float32)
    force = np.zeros_like(inside_positions, dtype=np.float32)

    r = inside_positions - particle
    r2 = np.sum(r**2, axis=1)
    check = r2 <= (2.0 * sigma)**2
    r2 = r2[check]

    r6 = r2**3
    r8 = r6 * r2 + alpha
    r6 += alpha

    calculated_force = (A / r8) * ((2 * (sigma**6) / r6) - 1)

    x_force = calculated_force * r[check, 0]
    y_force = calculated_force * r[check, 1]
    z_force = calculated_force * r[check, 2]

    particle_force[0] += np.sum(x_force)
    particle_force[1] += np.sum(y_force)
    particle_force[2] += np.sum(z_force)

    force[check, 0] -= x_force
    force[check, 1] -= y_force
    force[check, 2] -= z_force

    return particle_force, force

# @njit(parallel=True, fastmath=True, error_model="numpy")
def calc_force(particles, config):
    particles.force[:, :] = 0
    particles.force[:, 2] = g * particles.m
    shape_boxes = config.boxes.shape

    for j, k, l in product(range(shape_boxes[0]), range(shape_boxes[1]), range(shape_boxes[2])):
        config.boxes[j, k, l] = []

    x = (particles.positions[:, 0] // config.l_box).astype(np.int64)
    y = (particles.positions[:, 1] // config.l_box).astype(np.int64)
    z = (particles.positions[:, 2] // config.l_box).astype(np.int64)

    for i in range(particles.positions.shape[0]):
        config.boxes[x[i], y[i], z[i]].append(i)

    for i in prange(1, particles.positions.shape[0]):
        particle = particles.positions[i, :]
        i_s = range(max(0, x[i] - 1), min(shape_boxes[0], x[i] + 2))
        j_s = range(max(0, y[i] - 1), min(shape_boxes[1], y[i] + 2))
        k_s = range(max(0, z[i] - 1), min(shape_boxes[2], z[i] + 2))

        nearby_boxes = flatten(config.boxes, i_s, j_s, k_s)
        nearby_boxes = nearby_boxes[nearby_boxes < i]

        if len(nearby_boxes) == 0:
            continue

        neighbour_particles = particles.positions[nearby_boxes, :]
        result = calc_lennard_jones_force(particle, neighbour_particles, particles.A, particles.sigma)

        particles.force[i, :] += result[0]
        particles.force[nearby_boxes, :] += result[1]

    # Update particle velocities and positions based on calculated forces
    particles.v += (particles.force / particles.m) * config.delta_t
    particles.positions += particles.v * config.delta_t + (
        (particles.force / (2 * particles.m)) * config.delta_t**2
    )


# Function to handle wall interactions and temperature control
def wall_interactions(
    particles: particles_class,
    config: config_class,
    have_temp: bool = False,
    temp: float = 273.15,
):
    pressure = 0.0

    # Reflect particles hitting the walls
    check = particles.positions[:, 0] < 0
    particles.v[check, 0] = -particles.v[check, 0]
    particles.positions[check, 0] = 0
    pressure += np.sum(np.abs(particles.v[check, 0]))

    check = particles.positions[:, 0] > config.L[0]
    particles.v[check, 0] = -particles.v[check, 0]
    particles.positions[check, 0] = config.L[0]
    pressure += np.sum(np.abs(particles.v[check, 0]))

    check = particles.positions[:, 1] < 0
    particles.v[check, 1] = -particles.v[check, 1]
    particles.positions[check, 1] = 0
    pressure += np.sum(np.abs(particles.v[check, 1]))

    check = particles.positions[:, 1] > config.L[1]
    particles.v[check, 1] = -particles.v[check, 1]
    particles.positions[check, 1] = config.L[1]
    pressure += np.sum(np.abs(particles.v[check, 1]))

    check = particles.positions[:, 2] < 0
    particles.v[check, 2] = -particles.v[check, 2]
    particles.positions[check, 2] = 0
    pressure += np.sum(np.abs(particles.v[check, 2]))

    if have_temp:
        # Rescale velocities to achieve desired temperature
        norm = np.linalg.norm(particles.v[check, :], axis=-1)
        particles.make_scale(temp)
        speed = maxwell.ppf(np.random.rand(np.sum(check)), scale=particles.scale)
        particles.v[check, 0] = particles.v[check, 0] / norm * speed
        particles.v[check, 1] = particles.v[check, 1] / norm * speed
        particles.v[check, 2] = particles.v[check, 2] / norm * speed

    check = particles.positions[:, 2] > config.L[2]
    particles.v[check, 2] = -particles.v[check, 2]
    particles.positions[check, 2] = config.L[2]
    pressure += np.sum(np.abs(particles.v[check, 2]))

    # Calculate and return pressure
    pressure = pressure * particles.m / (config.area * config.delta_t)
    return pressure

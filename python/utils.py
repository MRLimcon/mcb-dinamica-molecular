import numpy as np
from scipy.stats import maxwell
from numba import njit, jit, prange
from itertools import product

# Constants
e_0 = 8.8541878128e-12
k_b = 1.380649e-23
g = -9.8

# Aliases for logical operations
np_or = np.logical_or
np_and = np.logical_and

# Configuration class
class config_class:
    L = [0, 0, 0]
    T = 273.15
    l_box = 1.
    delta_t = 0
    boxes = []
    area = 0

    def make_boxes(self, new_l: list[float]):
        # Calculate the amount of boxes
        amount = np.array([new_l[0] // self.l_box, new_l[1] // self.l_box, new_l[2] // self.l_box]).astype(int)

        # Create an array of lists
        self.boxes = np.empty((amount[0], amount[1], amount[2]), dtype=object)

        # Initialize each element as an empty list
        for j in range(amount[0]):
            for k in range(amount[1]):
                for l in range(amount[2]):
                    self.boxes[j, k, l] = []

    def __init__(self, L: list[float], T: float = 273.15, sigma: float = 1, delta_t: float = 0.05):
        self.L = L
        self.T = T
        self.l_box = 2*sigma
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

    def make_scale(self, T):
        # Calculate the scaling factor for particle velocity
        self.scale = np.sqrt(T * k_b / self.m)

    def __init__(self, N: int, L: list[float], mass: float = 1.164026113e-21, sigma: float = 1, T: float = 273.15):
        self.m = mass
        self.B = 6 * 4 * e_0 * (sigma ** 6)
        self.A = 12 * 4 * e_0 * (sigma ** 12)
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
        theta = np.random.rand(N) * 3.1415 * 2.
        phi = np.random.rand(N) * 3.1415
        avg_v = np.sqrt(8 * k_b * T / (np.pi * self.m))

        self.v[:, 0] = np.cos(theta) * np.sin(phi) * avg_v
        self.v[:, 1] = np.sin(theta) * np.sin(phi) * avg_v
        self.v[:, 2] = np.cos(phi) * avg_v

# Function to calculate Lennard-Jones force between particles
def calc_lennard_jones_force(inside_positions: np.ndarray, nearby_positions: np.ndarray, force: np.ndarray, A: float, B: float):
    for i in prange(inside_positions.shape[0]):
        # Calculate distance vectors between particles
        r = nearby_positions - inside_positions[i, :]
        normed_r = np.zeros_like(r)

        # Normalize the distance vectors
        norms = np.linalg.norm(r, axis=-1)
        normed_r[:, 0] = r[:, 0] / norms
        normed_r[:, 1] = r[:, 1] / norms
        normed_r[:, 2] = r[:, 2] / norms

        # Calculate Lennard-Jones forces
        calculated_force = (-A * (norms ** -13)) + (B * (norms ** -7))

        # Accumulate forces
        force[i, 0] += np.sum(calculated_force * normed_r[:, 0])
        force[i, 1] += np.sum(calculated_force * normed_r[:, 1])
        force[i, 2] += np.sum(calculated_force * normed_r[:, 2])

        if i >= 1:
            # Repeat the process for previously calculated particles
            r = inside_positions[:i, :] - inside_positions[i, :]
            normed_r = np.zeros_like(r)

            norms = np.linalg.norm(r, axis=-1)
            normed_r[:, 0] = r[:, 0] / norms
            normed_r[:, 1] = r[:, 1] / norms
            normed_r[:, 2] = r[:, 2] / norms

            calculated_force = (-A * (norms ** -13)) + (B * (norms ** -7))

            x_force = calculated_force * normed_r[:, 0]
            y_force = calculated_force * normed_r[:, 1]
            z_force = calculated_force * normed_r[:, 2]

            force[i, 0] += np.sum(x_force)
            force[i, 1] += np.sum(y_force)
            force[i, 2] += np.sum(z_force)
            force[:i, 0] += -x_force
            force[:i, 1] += -y_force
            force[:i, 2] += -z_force


@jit(parallel=True, fastmath=True)
def flatten(nearby_boxes):
    return [item for sublist in nearby_boxes for item in sublist]


# Function to calculate forces on particles
def calc_force(particles: particles_class, config: config_class):
    # Reset forces and apply gravitational force
    particles.force[:, :] = 0
    particles.force[:, 2] = g * particles.m

    for j, k, l in product(range(config.boxes.shape[0]), range(config.boxes.shape[1]), range(config.boxes.shape[2])):
        config.boxes[j, k, l] = []

    # Populate config.boxes with particle indices
    for i in prange(particles.positions.shape[0]):
        x = min(int(particles.positions[i, 0] // config.l_box), config.boxes.shape[0]-1)
        y = min(int(particles.positions[i, 1] // config.l_box), config.boxes.shape[1]-1)
        z = min(int(particles.positions[i, 2] // config.l_box), config.boxes.shape[2]-1)
        config.boxes[x, y, z].append(i)

    for i in prange(config.boxes.shape[0]):
        for j, k in product(range(config.boxes.shape[1]), range(config.boxes.shape[2])):
            inside_box_check = np.array(config.boxes[i, j, k])
            inside_box_check = inside_box_check[(inside_box_check >= 0) & (inside_box_check <= particles.positions.shape[0])]

            i_s = set(range(max(0, i-1), min(config.boxes.shape[0], i+2))).difference([i])
            j_s = set(range(max(0, j-1), min(config.boxes.shape[1], j+2))).difference([j])
            k_s = set(range(max(0, k-1), min(config.boxes.shape[2], k+2))).difference([k])

            nearby_boxes = config.boxes[list(i_s), list(j_s), list(k_s)]
            nearby_box_check = np.array(flatten(nearby_boxes), dtype=int)
            nearby_box_check = nearby_box_check[(nearby_box_check >= 0) & (nearby_box_check <= particles.positions.shape[0])]

            if inside_box_check.shape[0] > 0:
                calc_lennard_jones_force(
                    particles.positions[inside_box_check, :],
                    particles.positions[nearby_box_check, :],
                    particles.force[inside_box_check],
                    particles.A, particles.B
                )

    # Update particle velocities and positions based on calculated forces
    particles.v += (particles.force / particles.m) * config.delta_t
    particles.positions += particles.v * config.delta_t + ((particles.force / (2 * particles.m)) * config.delta_t ** 2)

# Function to handle wall interactions and temperature control
def wall_interactions(particles: particles_class, config: config_class, have_temp: bool = False, temp: float = 273.15):
    pressure = 0.

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

import numpy as np
from scipy.stats import maxwell
from numba import njit, jit

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
        # Create boxes in the simulation space
        box_x = np.arange(0, new_l[0], self.l_box)
        box_y = np.arange(0, new_l[1], self.l_box)
        box_z = np.arange(0, new_l[2], self.l_box)

        x, y, z = np.meshgrid(box_x, box_y, box_z)
        x, y, z = x.ravel(), y.ravel(), z.ravel()

        # Flatten the meshgrid to create boxes
        self.boxes = np.zeros([x.shape[0], 3])
        self.boxes[:, 0], self.boxes[:, 1], self.boxes[:, 2] = x, y, z

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
        theta = np.random.rand(N) * 3.1415
        phi = np.random.rand(N) * 3.1415 / 2.
        avg_v = np.sqrt(8 * k_b * T / (np.pi * self.m))

        self.v[:, 0] = np.cos(theta) * np.sin(phi) * avg_v
        self.v[:, 1] = np.sin(theta) * np.sin(phi) * avg_v
        self.v[:, 2] = np.cos(phi) * avg_v

# Function to calculate Lennard-Jones force between particles
def calc_lennard_jones_force(inside_positions: np.ndarray, nearby_positions: np.ndarray, force: np.ndarray, A: float, B: float):
    for i in range(inside_positions.shape[0]):
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

# Function to calculate forces on particles
def calc_force(particles: particles_class, config: config_class):
    delta_l = config.l_box

    # Reset forces and apply gravitational force
    particles.force[:, :] = 0
    particles.force[:, 2] = g * particles.m

    for i in np.arange(config.boxes.shape[0]):
        box = config.boxes[i]
        inside_box_check = (
            np_and(particles.positions[:, 0] >= box[0], particles.positions[:, 0] <= box[0] + delta_l) &
            np_and(particles.positions[:, 1] >= box[1], particles.positions[:, 1] <= box[1] + delta_l) &
            np_and(particles.positions[:, 2] >= box[2], particles.positions[:, 2] <= box[2] + delta_l)
        )

        nearby_box_check = (
            np_and(particles.positions[:, 0] >= box[0] - delta_l, particles.positions[:, 0] <= box[0] + 2 * delta_l) &
            np_and(particles.positions[:, 1] >= box[1] - delta_l, particles.positions[:, 1] <= box[1] + 2 * delta_l) &
            np_and(particles.positions[:, 2] >= box[2] - delta_l, particles.positions[:, 2] <= box[2] + 2 * delta_l)
        )

        nearby_box_check = np_and(nearby_box_check, np.logical_not(inside_box_check))

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

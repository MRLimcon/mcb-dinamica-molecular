import matplotlib.pyplot as plt
import utils

# Set the plotting style
plt.style.use(["ggplot", "fast"])

# Number of particles and simulation box dimensions
N_particles = 2000
L = [5, 5, 5]

# Initialize configuration and particle classes using utility module
consts = utils.config_class(L, sigma=1)
particles = utils.particles_class(N_particles, L, sigma=1)

# Initial plot of particle positions
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(
    particles.positions[:, 0], particles.positions[:, 1], particles.positions[:, 2]
)
ax.set_zlim(0, 5)
plt.show()

# Simulation loop
for iter in range(5000):
    print(iter)
    
    # Calculate forces and update particle positions and velocities
    utils.calc_force(particles, consts)
    
    # Handle wall interactions and optionally control temperature
    pressure = utils.wall_interactions(particles, consts, True, 1500)

    print(pressure)

    # Plot particle positions every 100 iterations
    if iter % 100 == 0:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            particles.positions[:, 0], particles.positions[:, 1], particles.positions[:, 2]
        )
        ax.set_zlim(0, 5)
        plt.show()

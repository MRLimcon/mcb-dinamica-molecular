using Random
using Statistics
using Plots
include("./utils2d.jl")

Random.seed!(123)  # Set a seed for reproducibility

println("Started")

""" 2d

N_particles = 1000
sigma = Float32(0.5)
dt = Float32(0.003)
T = Float32(0)
mass = Float32(1)
L = [4f0, 4f0, 4f0]
t_f = 5f0

"""

""" 3d

N_particles = 1000
sigma = Float32(0.2)
dt = Float32(0.01)
T = Float32(0)
mass = Float32(1)
L = [1f0, 1f0, 4f0]
t_f = 20f0

"""

N_particles = 1000
sigma = Float32(0.5)
dt = Float32(0.003)
T = Float32(0)
mass = Float32(1) # .164026113e-21)
L = [4.0f0, 4.0f0, 4.0f0]
t_f = 5.0f0
iterations = Int(div(t_f, dt))

particles = Utils.make_particles(N_particles, L, mass, sigma, T, true)
consts = Utils.make_config(L, T, sigma, dt)

plotted = scatter(
    particles.positions[:, 1],
    particles.positions[:, 2],
    # particles.positions[:, 3],
    xlabel = "X",
    ylabel = "Y",
    # zlabel = "Z",
    xlim = (0, L[1]),
    ylim = (0, L[2]),
    # zlim = (0, L[3]),
    # camera = (0, 15),
    markersize = 1,
    label = "Time: 0",
    legend = :topleft,
)
display(plotted)

for iter = 1:iterations
    println("$iter out of $iterations")

    global particles = Utils.calc_force(particles, consts)

    t = (iter - 1) * consts.delta_t
    if t < 1.7
        global pressure, particles = Utils.wall_interactions(particles, consts) # , true, 0.001f0)
    else
        global pressure, particles = Utils.wall_interactions(particles, consts)
    end

    println("Pressure: $pressure")

    camera_angle = t * 360.0
    plotted = scatter(
        particles.positions[:, 1],
        particles.positions[:, 2],
        # particles.positions[:, 3],
        xlabel = "X",
        ylabel = "Y",
        # zlabel = "Z",
        xlim = (0, L[1]),
        ylim = (0, L[2]),
        # zlim = (0, L[3]),
        # camera = (camera_angle, 15),
        markersize = 1,
        label = "Time: $t",
        legend = :topleft,
    )
    display(plotted)
end

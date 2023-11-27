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
sigma = Float32(0.25)
dt = Float32(0.005)
T = Float32(2)
mass = Float32(1) # .164026113e-21)
L = [4.0f0, 4.0f0, 4.0f0]
t_f = 20f0
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
    if t < 10
        global pressure, particles = Utils.wall_interactions(particles, consts) # 
    else
        global pressure, particles = Utils.wall_interactions(particles, consts, true, 1f0)
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

    display(plot!())
    """

    v2_val, v2, v_s, pdf = Utils.get_maxwell_dist(particles)
    dist = Utils.get_cin_energy(particles) ./ ((particles.m * particles.m) * size(particles.positions, 1))

    Plots.scatter([v2_val], [v2], label = "RMS Speed")
    Plots.plot!(v_s, pdf, label = "Maxwell Distribution")
    Plots.histogram!(dist, normalize=:pdf, label = "Histogram")
    display(plot!())
    """
end

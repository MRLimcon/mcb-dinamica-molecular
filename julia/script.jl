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


N_particles = 1000
sigma = Float32(0.2)
dt = Float32(0.01)
T = Float32(0)
mass = Float32(1)
L = [4.0f0, 4.0f0, 4.0f0]
t_f = 10.0f0



"""N_particles = 1000
sigma = Float32(0.25)
dt = Float32(0.005)
T = Float32(2)
mass = Float32(1) # .164026113e-21)
L = [4.0f0, 4.0f0, 4.0f0]
t_f = 20f0"""
iterations = Int(div(t_f, dt))

particles = Utils.make_particles(N_particles, L, mass, sigma, T, false)
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

range = 1:iterations

pressures = zeros(Float32, size(range))
temperatures = zeros(Float32, size(range))

for iter = 1:iterations
    println("$iter out of $iterations")

    global particles = Utils.calc_force(particles, consts, iter)

    t = (iter - 1) * consts.delta_t
    if t < 10
        global pressures[iter], particles = Utils.wall_interactions(particles, consts) # 
    else
        global pressures[iter], particles =
            Utils.wall_interactions(particles, consts, true, 1.0f0)
    end
    v2, Temp = Utils.get_v2_t(particles, consts.delta_t)
    temperatures[iter] = Temp
    pressure = pressures[iter]
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

    plot_pressure = plot(1:iter, pressures[1:iter], label = "Pressure")
    plot_temp = plot(1:iter, temperatures[1:iter], label = "Temperatures")

    v2_val, v2, v_s, pdf = Utils.get_maxwell_dist(particles, consts.delta_t)
    dist = sqrt.(vec(sum(particles.v .* particles.v, dims = 2)))

    Plots.scatter([v2], [v2_val], label = "RMS Speed")
    Plots.plot!(v_s, pdf, label = "Maxwell Distribution")
    plot_dist = Plots.histogram!(dist, normalize = :pdf, label = "Histogram")

    display(plot(plotted, plot_pressure, plot_temp, plot_dist, layout = (2, 2)))

end

bar_plot = bar(
    pressures,
    xlabel = "Iteration",
    ylabel = "Pressure",
    label = "Pressure",
    legend = :topleft,
)
display(bar_plot)
sleep(1)
bar_plot = bar(
    temperatures,
    xlabel = "Iteration",
    ylabel = "Temperature",
    label = "Temperature",
    legend = :topleft,
)
display(bar_plot)
sleep(1)

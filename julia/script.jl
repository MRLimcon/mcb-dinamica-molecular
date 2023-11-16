using Random
using Statistics
using Plots
include("./utils.jl")

Random.seed!(123)  # Set a seed for reproducibility

println("Started")

N_particles = 20000
L = [5.0f0, 5.0f0, 5.0f0]

particles = Utils.make_particles(N_particles, L, Float32(1.164026113e-21), Float32(0.1))
consts = Utils.make_config(L, Float32(273.15), Float32(0.1), Float32(0.1))


for iter in 1:500
    println(iter)
    Utils.calc_force!(particles, consts)
    pressure = Utils.wall_interactions!(particles, consts)# , true, 5000f0)

    println("Pressure: $pressure")

    t = (iter-1)*consts.delta_t
    plotted = scatter(
        particles.positions[:, 1], particles.positions[:, 2], particles.positions[:, 3],
        xlabel="X", ylabel="Y", zlabel="Z", xlim=(0, L[1]), ylim=(0, L[2]), zlim=(0, L[3]),
        markersize=1, label="Time: $t", legend=:topleft
    )
    display(plotted)
end

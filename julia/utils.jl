module Utils
using Random
using PyCall
# using Conda

# Conda.add("scipy")
scipy = pyimport("scipy")
maxwell = scipy.stats.maxwell

const e_0 = Float32(0.05)# Float32(8.8541878128e-12)
const k_b = Float32(1) # .380649e-23)
const g = Float32(-1)
const alpha = 1e-8
const damping = 0

mutable struct Config
    L::Vector{Float32}
    T::Float32
    l_box::Float32
    delta_t::Float32
    boxes::Array{Vector{Int}}
    area::Float32
end

function make_boxes(new_l::Vector{Float32}, l_box::Float32)
    amount =
        Int.([
            div(new_l[1], l_box, RoundDown),
            div(new_l[2], l_box, RoundDown),
            div(new_l[3], l_box, RoundDown),
        ])

    boxes = Array{Vector{Int}}(undef, amount[1] + 1, amount[2] + 1, amount[3] + 1)

    for j = 1:amount[1]+1, k = 1:amount[2]+1, l = 1:amount[3]+1
        boxes[j, k, l] = Vector{Int}()
    end

    return boxes
end

function make_config(
    L::Vector{Float32},
    T::Float32 = Float32(273.15),
    sigma::Float32 = Float32(1.0),
    delta_t::Float32 = Float32(0.05),
)
    l_box = 2.0 * sigma
    boxes = make_boxes(L, Float32(l_box))

    return Config(
        L,
        T,
        l_box,
        delta_t,
        boxes,
        2 * L[1] * L[2] + 2 * L[1] + L[3] + 2 * L[2] * L[3],
    )
end

mutable struct Particles
    v::Matrix{Float32}
    positions::Matrix{Float32}
    dims::Vector{Int}
    force::Matrix{Float32}
    m::Float32
    A::Float32
    B::Float32
    scale::Float32
    sigma::Float32
end

function generate_grid_particles(n::Vector{Int64}, L::Vector{Float32})
    x = (collect(1:n[1]) .- 0.5) .* L[1] ./ n[1]
    y = (collect(1:n[2]) .- 0.5) .* L[2] ./ n[2]
    z = (collect(1:n[3]) .- 0.5) .* L[3] ./ n[3]

    xs = [xx for xx in x, yy in y, zz in z]
    ys = [yy for xx in x, yy in y, zz in z]
    zs = [zz for xx in x, yy in y, zz in z]

    particles = hcat(xs[:], ys[:], zs[:])

    return particles
end

function make_particles(
    N::Int,
    L::Vector{Float32},
    mass::Float32 = Float32(1.164026113e-21),
    sigma::Float32 = Float32(1),
    T::Float32 = Float32(273.15),
    is_random::Bool = true,
)
    A = Float32(12) * Float32(4) * e_0 * (sigma^12)
    B = Float32(6) * Float32(4) * e_0 * (sigma^6)
    scale = sqrt(T * k_b / mass)

    if is_random
        dims = [N, 3]
        positions = zeros(Float32, dims...)
        positions[:, 1] .= rand(N) * L[1]
        positions[:, 2] .= rand(N) * L[2]
        positions[:, 3] .= rand(N) * L[3]
    else
        N_s = [cbrt(N), cbrt(N), cbrt(N)]

        # Round up to the nearest integer
        N_s = ceil.(Int, N_s)

        # Call the function to generate grid particles
        positions = generate_grid_particles(N_s, L)
    end

    n_particles = size(positions, 1)
    dims = [n_particles, 3]

    force = zeros(Float32, dims...)
    v = zeros(Float32, dims...)

    theta = rand(n_particles) * π * Float32(2.0)
    phi = rand(n_particles) * π
    avg_v = sqrt(Float32(8) * k_b * T / (π * mass))

    v[:, 1] .= cos.(theta) .* sin.(phi) .* avg_v
    v[:, 2] .= sin.(theta) .* sin.(phi) .* avg_v
    v[:, 3] .= cos.(phi) .* avg_v

    return Particles(v, positions, dims, force, mass, A, B, scale, sigma)
end


function calc_lennard_jones_force(
    particle::Vector{Float32},
    inside_positions::Matrix{Float32},
    A::Float32,
    sigma::Float32,
)
    n_inside = size(inside_positions, 1)

    particle_force = zeros(Float32, size(inside_positions, 2))

    force = zeros(Float32, size(inside_positions)...)

    r = zeros(Float32, n_inside, size(inside_positions, 2))

    for j = 1:size(inside_positions, 1)
        r[j, :] .= particle .- inside_positions[j, :]
    end
    r2 = sum(r .* r, dims = 2)
    check = vec(r2 .<= ((2.0 * sigma)^2))
    r2 = r2[check]
    r6 = r2 .* r2 .* r2
    r8 = (r6 .* r2) .+ alpha
    r6 .+= alpha

    calculated_force = (A ./ r8) .* ((2 .* (sigma^6) ./ r6) .- 1)

    x_force = calculated_force .* r[check, 1]
    y_force = calculated_force .* r[check, 2]
    z_force = calculated_force .* r[check, 3]

    particle_force[1] += sum(x_force)
    particle_force[2] += sum(y_force)
    particle_force[3] += sum(z_force)
    force[check, 1] .-= x_force
    force[check, 2] .-= y_force
    force[check, 3] .-= z_force

    return particle_force, force
end


function calc_force(particles::Particles, config::Config)
    particles.force[:, :] .= 0
    particles.force[:, 3] .= g * particles.m
    shape_boxes = [size(config.boxes, 1), size(config.boxes, 2), size(config.boxes, 3)]

    @sync Threads.@threads for j = 1:shape_boxes[1]
        for k = 1:shape_boxes[2], l = 1:shape_boxes[3]
            config.boxes[j, k, l] = Vector{Int}()
        end
    end


    x = Int.(div.(particles.positions[:, 1], config.l_box)) .+ 1
    y = Int.(div.(particles.positions[:, 2], config.l_box)) .+ 1
    z = Int.(div.(particles.positions[:, 3], config.l_box)) .+ 1
    for i = 1:size(particles.positions, 1)
        push!(config.boxes[x[i], y[i], z[i]], i)
    end


    @sync Threads.@threads for i = 2:size(particles.positions, 1)
        particle = particles.positions[i, :]

        i_s = max(1, x[i] - 1):min(shape_boxes[1], x[i] + 1)
        j_s = max(1, y[i] - 1):min(shape_boxes[2], y[i] + 1)
        k_s = max(1, z[i] - 1):min(shape_boxes[2], z[i] + 1)

        nearby_boxes =
            filter(box -> box < i, collect(Iterators.flatten(config.boxes[i_s, j_s, k_s])))
        if isempty(nearby_boxes)
            continue
        end
        neighbour_particles = particles.positions[nearby_boxes, :]

        result = calc_lennard_jones_force(
            particle,
            neighbour_particles,
            particles.A,
            particles.sigma,
        )
        particles.force[i, :] .+= result[1]
        particles.force[nearby_boxes, :] .+= result[2]
    end

    particles.v .+= (particles.force ./ particles.m) .* config.delta_t
    particles.positions .+=
        particles.v .* config.delta_t .+
        (((particles.force) ./ (2 * particles.m)) .* config.delta_t^2)

    return particles
end

function wall_interactions(
    particles::Particles,
    config::Config,
    have_temp::Bool = false,
    temp::Float32 = Float32(273.15),
)
    pressure::Float32 = Float32(0.0)

    check = particles.positions[:, 1] .<= 0
    particles.v[check, 1] .= -particles.v[check, 1]
    particles.positions[check, 1] .= 0
    pressure += 2 * sum(abs.(particles.v[check, 1]))

    check = particles.positions[:, 1] .>= config.L[1]
    particles.v[check, 1] .= -particles.v[check, 1]
    particles.positions[check, 1] .= config.L[1]
    pressure += 2 * sum(abs.(particles.v[check, 1]))

    check = particles.positions[:, 2] .<= 0
    particles.v[check, 2] .= -particles.v[check, 2]
    particles.positions[check, 2] .= 0
    pressure += 2 * sum(abs.(particles.v[check, 2]))

    check = particles.positions[:, 2] .>= config.L[2]
    particles.v[check, 2] .= -particles.v[check, 2]
    particles.positions[check, 2] .= config.L[2]
    pressure += 2 * sum(abs.(particles.v[check, 2]))

    check = particles.positions[:, 3] .<= 0
    particles.v[check, 3] .= -particles.v[check, 3]
    particles.positions[check, 3] .= 0
    pressure += 2 * sum(abs.(particles.v[check, 3]))

    if have_temp
        norm = vec(sqrt.(sum(particles.v[check, :] .* particles.v[check, :], dims = 2)))
        particles.scale = sqrt(temp * k_b / particles.m)
        speed = maxwell.ppf(rand(sum(check)), scale = particles.scale)
        particles.v[check, 1] .= particles.v[check, 1] ./ norm .* speed
        particles.v[check, 2] .= particles.v[check, 2] ./ norm .* speed
        particles.v[check, 3] .= particles.v[check, 3] ./ norm .* speed
    end

    check = particles.positions[:, 3] .>= config.L[3]
    particles.v[check, 3] .= -particles.v[check, 3]
    particles.positions[check, 3] .= config.L[3]
    pressure += 2 * sum(abs.(particles.v[check, 3]))

    # Calculate and return pressure
    pressure = pressure * particles.m / (config.area * config.delta_t)

    return pressure, particles
end
end

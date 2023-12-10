module Utils
using Random
using PyCall
# using Conda

# Conda.add("scipy")
scipy = pyimport("scipy")
maxwell = scipy.stats.maxwell

const e_0 = Float32(0.00005)# Float32(8.8541878128e-12)
const k_b = Float32(1) # .380649e-23)
const g = Float32(-0.5)
const alpha = 1e-10
const damping = 0.5f0

mutable struct Config
    L::Vector{Float32}
    T::Float32
    l_box::Float32
    delta_t::Float32
    boxes::Array{Vector{Int}}
    new_boxes::Array{Vector{Int}}
    area::Float32
end

function make_boxes(new_l::Vector{Float32}, l_box::Float32)
    amount = Int.([div(new_l[1], l_box, RoundDown), div(new_l[2], l_box, RoundDown)])

    boxes = Array{Vector{Int}}(undef, amount[1] + 1, amount[2] + 1)

    for j = 1:amount[1]+1, k = 1:amount[2]+1
        boxes[j, k] = Vector{Int}()
    end

    return boxes
end

function make_config(
    L::Vector{Float32},
    T::Float32 = Float32(273.15),
    sigma::Float32 = Float32(1.0),
    delta_t::Float32 = Float32(0.05),
)
    l_box = 2.5 * sigma
    boxes = make_boxes(L, Float32(l_box))
    new_boxes = make_boxes(L, Float32(l_box))

    area = 2 * L[1] + 2 * L[2]

    return Config(L, T, l_box, delta_t, boxes, new_boxes, area)
end

mutable struct Particles
    v::Matrix{Float32}
    positions::Matrix{Float32}
    new_positions::Matrix{Float32}
    last_positions::Matrix{Float32}
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

    xs = [xx for xx in x, yy in y]
    ys = [yy for xx in x, yy in y]

    particles = hcat(xs[:], ys[:])

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
    A = Float32(24) * e_0 * (sigma^12)
    B = Float32(6) * Float32(4) * e_0 * (sigma^6)
    scale = sqrt(T * k_b / mass)

    if is_random
        dims = [N, 2]
        positions = zeros(Float32, dims...)
        positions[:, 1] .= rand(N) * L[1]
        positions[:, 2] .= rand(N) * L[2]
    else
        N_s = [sqrt(N), sqrt(N)]

        # Round up to the nearest integer
        N_s = ceil.(Int, N_s)

        # Call the function to generate grid particles
        positions = generate_grid_particles(N_s, L)
    end

    n_particles = size(positions, 1)
    dims = [n_particles, 2]

    last_positions = zeros(Float32, dims...)
    new_positions = zeros(Float32, dims...)
    force = zeros(Float32, dims...)
    v = zeros(Float32, dims...)

    theta = rand(n_particles) * 3.1415 * Float32(2.0)
    phi = rand(n_particles) * 3.1415
    avg_v = sqrt(Float32(8) * k_b * T / (π * mass))

    v[:, 1] .= cos.(theta) .* avg_v
    v[:, 2] .= sin.(theta) .* avg_v

    return Particles(
        v,
        positions,
        new_positions,
        last_positions,
        dims,
        force,
        mass,
        A,
        B,
        scale,
        sigma,
    )
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
    check = vec(r2 .<= ((2.5 * sigma)^2))
    r2 = r2[check]
    r6 = r2 .* r2 .* r2
    r8 = (r6 .* r2) .+ alpha
    r6 .+= alpha

    calculated_force = (A ./ r8) .* ((2 .* (sigma^6) ./ r6) .- 1)

    x_force = calculated_force .* r[check, 1]
    y_force = calculated_force .* r[check, 2]

    particle_force[1] += sum(x_force)
    particle_force[2] += sum(y_force)
    force[check, 1] .-= x_force
    force[check, 2] .-= y_force

    return particle_force, force
end


function calc_force(particles::Particles, config::Config, iteration::Int)
    particles.force[:, :] .= 0
    particles.force[:, 2] .= g * particles.m
    shape_boxes = [size(config.boxes, 1), size(config.boxes, 2)]

    @sync Threads.@threads for j = 1:shape_boxes[1]
        for k = 1:shape_boxes[2]
            config.new_boxes[j, k] = Vector{Int}()
            config.boxes[j, k] = Vector{Int}()
        end
    end


    x = Int.(div.(particles.positions[:, 1], config.l_box)) .+ 1
    y = Int.(div.(particles.positions[:, 2], config.l_box)) .+ 1
    for i = 1:size(particles.positions, 1)
        push!(config.new_boxes[x[i], y[i]], i)
    end

    @sync Threads.@threads for j = 1:shape_boxes[1]
        for k = 1:shape_boxes[2]
            i_s = max(1, j - 1):min(shape_boxes[1], j + 1)
            j_s = max(1, k - 1):min(shape_boxes[2], k + 1)
            config.boxes[j, k] = collect(Iterators.flatten(config.new_boxes[i_s, j_s]))
        end
    end


    @sync Threads.@threads for i = 2:size(particles.positions, 1)
        particle = particles.positions[i, :]

        nearby_indices =
            filter(particle_index -> particle_index < i, config.boxes[x[i], y[i]])
        neighbour_particles = particles.positions[nearby_indices, :]
        if isempty(neighbour_particles)
            continue
        end

        result = calc_lennard_jones_force(
            particle,
            neighbour_particles,
            particles.A,
            particles.sigma,
        )
        particles.force[i, :] .+= result[1]
        particles.force[nearby_indices, :] .+= result[2]
    end


    if iteration == 1
        particles.new_positions .=
            particles.positions .+ particles.v .* config.delta_t .+
            (((particles.force) ./ (2 * particles.m)) .* config.delta_t^2)
    else
        particles.v .= (particles.positions .- particles.last_positions) ./ config.delta_t
        particles.force .-= particles.v .* damping
        particles.new_positions .=
            particles.positions .+ particles.v .* config.delta_t .+
            (((particles.force) ./ (particles.m)) .* config.delta_t^2)
    end

    particles.last_positions .= particles.positions
    particles.positions .= particles.new_positions

    return particles
end

function wall_interactions(
    particles::Particles,
    config::Config,
    have_temp::Bool = false,
    temp::Float32 = Float32(273.15),
)
    pressure::Float32 = Float32(0.0)

    @sync Threads.@threads for i = 1:size(particles.positions, 1)
        if particles.positions[i, 1] <= 0
            v =
                (particles.positions[i, 1] - particles.last_positions[i, 1]) /
                config.delta_t
            particles.positions[i, 1] = 0
            particles.last_positions[i, 1] = v * config.delta_t + particles.positions[i, 1]
            pressure += 2 * sum(abs(v))
        elseif particles.positions[i, 1] >= config.L[1]
            v =
                (particles.positions[i, 1] - particles.last_positions[i, 1]) /
                config.delta_t
            particles.positions[i, 1] = config.L[1]
            particles.last_positions[i, 1] = v * config.delta_t + config.L[1]
            pressure += 2 * sum(abs(v))
        end

        if particles.positions[i, 2] <= 0
            v =
                (particles.positions[i, 2] - particles.last_positions[i, 2]) /
                config.delta_t
            particles.positions[i, 2] = 0
            particles.last_positions[i, 2] = v * config.delta_t + particles.positions[i, 2]
            pressure += 2 * sum(abs(v))

            if have_temp
                norm = sqrt(sum(particles.v .* particles.v, dims = 2))
                particles.scale = sqrt(temp * k_b / particles.m)
                speed = maxwell.ppf(rand(1), scale = particles.scale)
                v = zeros(Float32, 1, 2)
                v[1] =
                    (particles.positions[i, 1] - particles.last_positions[i, 1]) /
                    config.delta_t / norm * speed
                v[2] =
                    (particles.positions[i, 2] - particles.last_positions[i, 2]) /
                    config.delta_t / norm * speed
                particles.last_positions[i, 1] =
                    v[1] * config.delta_t + particles.positions[i, 1]
                particles.last_positions[i, 2] =
                    v[2] * config.delta_t + particles.positions[i, 2]
            else
                particles.last_positions[i, 2] =
                    v * config.delta_t + particles.positions[i, 2]
            end
        elseif particles.positions[i, 2] >= config.L[2]
            v =
                (particles.positions[i, 2] - particles.last_positions[i, 2]) /
                config.delta_t
            particles.positions[i, 2] = config.L[2]
            particles.last_positions[i, 2] = v * config.delta_t + config.L[2]
            pressure += 2 * sum(abs(v))
        end
    end

    # Calculate and return pressure
    pressure = pressure * particles.m / (config.area * config.delta_t)

    return pressure, particles
end

function get_cin_energy(particles::Particles)
    u = particles.m .* vec(sum(particles.v .* particles.v, dims = 2))
    return u
end

function get_v2_t(particles::Particles, dt::Float32)
    v2 = vec(sum(particles.v .* particles.v, dims = 2))
    v2_avg = sum(v2) / size(v2, 1)
    t = (v2_avg * particles.m) / (3 * k_b)
    return v2_avg, t
end

function get_maxwell_dist(particles::Particles, dt::Float32)
    v2, t = get_v2_t(particles, dt)
    particles.scale = sqrt(t * k_b / particles.m)
    max_val = maxwell.ppf(0.999, scale = particles.scale)
    if isnan(max_val)
        max_val = 1.0f0
    end
    steps = 0.001f0

    v_s = collect(Float32, 0:steps:max_val)

    pdf = maxwell.pdf(v_s, scale = particles.scale)

    v2_val = maxwell.pdf(sqrt(v2), scale = particles.scale)

    return v2_val, sqrt(v2), v_s, pdf
end

end

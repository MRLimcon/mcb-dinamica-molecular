module Utils
using Random
using PyCall
# using Conda

# Conda.add("scipy")
scipy = pyimport("scipy")
maxwell = scipy.stats.maxwell

const e_0 = Float32(8.8541878128e-12)
const k_b = Float32(1.380649e-23)
const g = Float32(-9.8)

mutable struct Config
    L::Vector{Float32}
    T::Float32
    l_box::Float32
    delta_t::Float32
    boxes::Array{Vector{Int}}
    area::Float32
end

function make_boxes(new_l::Vector{Float32}, l_box::Float32)
    amount = Int.([div(new_l[1], l_box, RoundDown), div(new_l[2], l_box, RoundDown), div(new_l[3], l_box, RoundDown)])

    boxes = Array{Vector{Int}}(undef, amount[1]+1, amount[2]+1, amount[3]+1)

    for j in 1:amount[1]+1, k in 1:amount[2]+1, l in 1:amount[3]+1
        boxes[j, k, l] = Vector{Int}()
    end

    return boxes
end

function make_config(L::Vector{Float32}, T::Float32 = Float32(273.15), sigma::Float32 = Float32(1.0), delta_t::Float32 = Float32(0.05))
    l_box = 2.0 * sigma
    boxes = make_boxes(L, Float32(l_box))
    return Config(L, T, l_box, delta_t, boxes, 2 * L[1] * L[2] + 2 * L[1] + L[3] + 2 * L[2] * L[3])
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
end

function make_particles(N::Int, L::Vector{Float32}, mass::Float32 = Float32(1.164026113e-21), sigma::Float32 = Float32(1), T::Float32 = Float32(273.15))
    A = Float32(12) * Float32(4) * e_0 * (sigma^12)
    B = Float32(6) * Float32(4) * e_0 * (sigma^6)
    scale = sqrt(T * k_b / mass)

    dims = [N, 3]

    force = zeros(Float32, dims...)
    positions = zeros(Float32, dims...)
    positions[:, 1] .= rand(N) * L[1]
    positions[:, 2] .= rand(N) * L[2]
    positions[:, 3] .= rand(N) * L[3]

    v = zeros(Float32, dims...)

    theta = rand(N) * 3.1415
    phi = rand(N) * 3.1415 / Float32(2.0)
    avg_v = sqrt(Float32(8) * k_b * T / (π * mass))

    v[:, 1] .= cos.(theta) .* sin.(phi) .* avg_v
    v[:, 2] .= sin.(theta) .* sin.(phi) .* avg_v
    v[:, 3] .= cos.(phi) .* avg_v

    return Particles(v, positions, dims, force, mass, A, B, scale)
end


function calc_lennard_jones_force!(inside_positions::Matrix{Float32}, nearby_positions::Matrix{Float32}, force::Matrix{Float32}, A::Float32, B::Float32)
    n_nearby = size(nearby_positions, 1)
    n_inside = size(inside_positions, 1)

    n = max(n_nearby, n_inside)

    r = zeros(Float32, n, size(nearby_positions, 2))
    for i in 1:n_inside
        if n_nearby > 0
            for j in 1:n_nearby
                r[j, :] .= nearby_positions[j, :] .- inside_positions[i, :]
            end
            r2 = r[1:n_nearby, 1].*r[1:n_nearby, 1] .+ r[1:n_nearby, 2].*r[1:n_nearby, 2] .+ r[1:n_nearby, 3].*r[1:n_nearby, 3]
            norms = sqrt.(r2)
            r4 = r2 .* r2
            r6 = r4 .* r2
            r7 = r6 .* norms
            r13 = r6 .* r7

            normed_r = r[1:n_nearby, :] ./ norms

            calculated_force = (B ./ r7) .- (A ./ r13)

            force[i, 1] += sum(calculated_force .* normed_r[:, 1])
            force[i, 2] += sum(calculated_force .* normed_r[:, 2])
            force[i, 3] += sum(calculated_force .* normed_r[:, 3])
        end

        if i >= 2
            for j in 1:i-1
                r[j, :] .= inside_positions[j, :] .- inside_positions[i, :]
            end
            r2 = r[1:i-1, 1].*r[1:i-1, 1] .+ r[1:i-1, 2].*r[1:i-1, 2] .+ r[1:i-1, 3].*r[1:i-1, 3]
            norms = sqrt.(r2)
            r4 = r2 .* r2
            r6 = r4 .* r2
            r7 = r6 .* norms
            r13 = r6 .* r7

            normed_r = r[1:i-1, :] ./ norms

            calculated_force = (B ./ r7) .- (A ./ r13)

            x_force = calculated_force .* normed_r[:, 1]
            y_force = calculated_force .* normed_r[:, 2]
            z_force = calculated_force .* normed_r[:, 3]

            force[i, :] .+= [sum(x_force), sum(y_force), sum(z_force)]
            force[1:i-1, 1] .+= -x_force
            force[1:i-1, 2] .+= -y_force
            force[1:i-1, 3] .+= -z_force
        end
    end
end


function calc_force!(particles::Particles, config::Config)
    particles.force[:, :] .= 0
    particles.force[:, 3] .= g * particles.m

    
    for j in 1:size(config.boxes, 1), k in 1:size(config.boxes, 2), l in 1:size(config.boxes, 3)
        config.boxes[j, k, l] = Vector{Int}()
    end

    @sync Threads.@threads for i in 1:size(particles.positions, 1)
        x = Int(div(particles.positions[i, 1], config.l_box)) + 1
        y = Int(div(particles.positions[i, 2], config.l_box)) + 1
        z = Int(div(particles.positions[i, 3], config.l_box)) + 1
        push!(config.boxes[x, y, z], i)
    end

    shape_boxes = [size(config.boxes, 1), size(config.boxes, 2), size(config.boxes, 3)]
    @sync Threads.@threads for i in 1:shape_boxes[1]
        for j in 1:shape_boxes[2], k in 1:shape_boxes[3]
            inside_box_check = config.boxes[i, j, k]
            inside_box_check = inside_box_check[1 .<= inside_box_check .<= size(particles.positions, 1)]

            i_s = setdiff(max(1, i-1):min(shape_boxes[1], i+1), i)
            j_s = setdiff(max(1, j-1):min(shape_boxes[2], j+1), j)
            k_s = setdiff(max(1, k-1):min(shape_boxes[3], k+1), k)
            
            nearby_box_check = collect(Iterators.flatten(config.boxes[i_s, j_s, k_s]))
            nearby_box_check = nearby_box_check[1 .<= nearby_box_check .<= size(particles.positions, 1)]

            if size(inside_box_check, 1) > 0
                calc_lennard_jones_force!(
                    particles.positions[inside_box_check, :],
                    particles.positions[nearby_box_check, :],
                    particles.force[inside_box_check, :],
                    particles.A,
                    particles.B,
                )
            end
        end
    end

    particles.v .+= (particles.force ./ particles.m) .* config.delta_t
    particles.positions .+= particles.v .* config.delta_t + ((particles.force ./ (2 * particles.m)) .* config.delta_t^2)
end

function wall_interactions!(particles::Particles, config::Config, have_temp::Bool = false, temp::Float32 = Float32(273.15), pressure::Float32 = Float32(0.))
    check = particles.positions[:, 1] .< 0
    particles.v[check, 1] .= -particles.v[check, 1]
    particles.positions[check, 1] .= 0
    pressure += 2*sum(abs.(particles.v[check, 1]))

    check = particles.positions[:, 1] .> config.L[1]
    particles.v[check, 1] .= -particles.v[check, 1]
    particles.positions[check, 1] .= config.L[1]
    pressure += 2*sum(abs.(particles.v[check, 1]))

    check = particles.positions[:, 2] .< 0
    particles.v[check, 2] .= -particles.v[check, 2]
    particles.positions[check, 2] .= 0
    pressure += 2*sum(abs.(particles.v[check, 2]))

    check = particles.positions[:, 2] .> config.L[2]
    particles.v[check, 2] .= -particles.v[check, 2]
    particles.positions[check, 2] .= config.L[2]
    pressure += 2*sum(abs.(particles.v[check, 2]))

    check = particles.positions[:, 3] .< 0
    particles.v[check, 3] .= -particles.v[check, 3]
    particles.positions[check, 3] .= 0
    pressure += 2*sum(abs.(particles.v[check, 3]))

    if have_temp
        norm = vec(sqrt.(sum(particles.v[check, :] .^ 2, dims=2)))
        particles.scale = sqrt(temp * k_b / particles.m)
        speed = maxwell.ppf(rand(sum(check)), scale=particles.scale)
        particles.v[check, 1] .= particles.v[check, 1] ./ norm .* speed
        particles.v[check, 2] .= particles.v[check, 2] ./ norm .* speed
        particles.v[check, 3] .= particles.v[check, 3] ./ norm .* speed
    end

    check = particles.positions[:, 3] .> config.L[3]
    particles.v[check, 3] .= -particles.v[check, 3]
    particles.positions[check, 3] .= config.L[3]
    pressure += 2*sum(abs.(particles.v[check, 3]))

    # Calculate and return pressure
    pressure = pressure * particles.m / (config.area * config.delta_t)

    return pressure
end
end
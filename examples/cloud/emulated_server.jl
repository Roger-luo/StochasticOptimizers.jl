include("server.jl")
include("emulator_patch.jl")

using RydbergEmulator
using LightGraphs

# finding the exact solution
# julia> dev https://github.com/GiggleLiu/EliminateGraphs.jl
using EliminateGraphs
exact_solve_mis(g::AbstractGraph) = mis2(EliminateGraph(adjacency_matrix(g)))

# generate random atom positions
atoms = rand_atoms(10, 1.5)
graph = unit_disk_graph(atoms, 1.0)
cfg = EmulatorConfig(atoms, graph, 50)
display(cfg)

# faithful simulate one experiment
ϕs = rand(5)*2π
ts = rand(5)
res = faithful_simulate(cfg, ϕs, ts)
display(res)

println("The maximum independent set size is $(exact_solve_mis(graph)).")

# start up a service at port 2020.
run_server(cfg; port=2020, output_folder=joinpath(@__DIR__, "data"))

#=
# To implement a new backend

struct DeviceConfig{GT} <: ExperimentConfig
    atoms::Vector{Atom2D{Float64}}
    graph::GT
    nshots::Int
end
nshots(c::DeviceConfig) = c.nshots
nbits(c::DeviceConfig) = nv(c.graph)

function run_experiment(config::EmulatorConfig, params)
    ϕs = params[:,1]
    ts = params[:,2]
    result = zeros(Int, nshots(config), nbits(config))

    # run experiment
    ...

    return result
end

=#

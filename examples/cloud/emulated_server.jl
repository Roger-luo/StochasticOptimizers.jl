include("server.jl")

using RydbergEmulator
using LightGraphs

using EliminateGraphs
exact_solve_mis(g::AbstractGraph) = mis2(EliminateGraph(adjacency_matrix(g)))

struct EmulatorConfig{GT} <: ExperimentConfig
    graph::GT
    nshots::Int
end
nshots(c::EmulatorConfig) = c.nshots
nbits(c::EmulatorConfig) = nv(c.graph)

function run_experiment(config::EmulatorConfig, params)
    ϕs = params[:,1]
    ts = params[:,2]
    res = qaoa_on_graph(config.graph, ϕs, ts) |> measure(;nshots=nshots(config))
    packed = zeros(Int, nshots(config), nbits(config))
    for i=1:nshots(config)
        for j=1:nbits(config)
            packed[i,j] = res[i][j]
        end
    end
    return packed
end

graph = rand_unit_disk_graph(10, 1.5)
@show exact_solve_mis(graph)
run_server(EmulatorConfig(graph, 50);
    port=2020, output_folder=joinpath(@__DIR__, "data"))

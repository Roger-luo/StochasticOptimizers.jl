using RydbergEmulator, Viznet, Compose
using RydbergEmulator: Atom2D
using LightGraphs
using BitBasis

viz_atoms(atoms::Vector{<:Atom2D}) = viz_atoms(atoms, unit_disk_graph(atoms))

function convert_locs(atoms::Vector{<:Atom2D})
    xmin = minimum(x->x.loc[1], atoms)
    ymin = minimum(x->x.loc[2], atoms)
    xmax = maximum(x->x.loc[1], atoms)
    ymax = maximum(x->x.loc[2], atoms)
    scale = max(xmax-ymin, ymax-ymin)*1.1
    newlocs = map(x-> (x.loc .- (xmin, ymin)) ./ scale .+ 0.05, atoms)
end

function viz_atoms(atoms::Vector{<:Atom2D}, graph::SimpleGraph)
    line_style=compose(bondstyle(:default), stroke("black"))
    node_style=compose(context(), nodestyle(:default), stroke("black"), fill("white"), linewidth(0.5mm))
    text_style=textstyle(:default)
    newlocs = convert_locs(atoms)
    canvas() do
        for (i, node) in enumerate(newlocs)
            node_style >> node
            text_style >> (node, "$i")
        end
        for eg in edges(graph)
            line_style >> (newlocs[eg.src], newlocs[eg.dst])
        end
    end
end

function viz_config(atoms::Vector{<:Atom2D}, graph::SimpleGraph, config::BitStr)
    g1 = viz_atoms(atoms, graph, )
    rb =compose(context(), nodestyle(:default), stroke("black"), fill("red"), linewidth(0.5mm), fillopacity(0.3))
    newlocs = convert_locs(atoms)
    g2 = canvas() do
        for (b, loc) in zip(config, newlocs)
            b == 1 && rb >> loc
        end
    end
    compose(context(), g2, g1)
end

function rand_bitstr64(nbit::Int)
    T = BitStr64{nbit}
    rand(typemin(T):typemax(T))
end
Base.rand(::Type{T}) where T<:BitStr = rand(typemin(T):typemax(T))

struct EmulatorConfig{GT} <: ExperimentConfig
    atoms::Vector{Atom2D{Float64}}
    graph::GT
    nshots::Int
end
nshots(c::EmulatorConfig) = c.nshots
nbits(c::EmulatorConfig) = nv(c.graph)
Base.display(em::EmulatorConfig) = display(viz_atoms(em.atoms, em.graph))

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

struct SimulationRes{EC<:ExperimentConfig,RT}
    cfg::EC
    res::RT
end
Base.display(em::SimulationRes) = display(viz_config(em.cfg.atoms, em.cfg.graph, em.res))

function faithful_simulate(cfg::EmulatorConfig, ϕs::AbstractVector, ts::AbstractVector)
    res = qaoa_on_graph(cfg.graph, ϕs, ts) |> measure!
    return SimulationRes(cfg, res)
end

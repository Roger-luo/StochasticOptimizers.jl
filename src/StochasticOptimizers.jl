module StochasticOptimizers

using NiLang
using LsqFit
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using Evolutionary

export CMAES, optimize

include("mgd.jl")
include("modified_mgd.jl")
include("spsa.jl")

end # module
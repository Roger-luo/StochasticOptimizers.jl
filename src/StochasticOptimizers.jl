module StochasticOptimizers

using NiLang
using LsqFit
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using Evolutionary
using Evolutionary: AbstractOptimizer

import Evolutionary: CMAES, optimize, update_state!

export CMAES, optimize, update_state!

include("mgd.jl")
include("modified_mgd.jl")
include("spsa.jl")

end # module

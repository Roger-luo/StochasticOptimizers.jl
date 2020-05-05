module StochasticOptimizers

using NiLang
using LsqFit
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using Evolutionary
using Evolutionary: AbstractOptimizer
using Distributions
using ProgressMeter
import Zygote

import Evolutionary: CMAES, optimize, update_state!

export CMAES, optimize, update_state!
export PolicyGrad, ADAGrad, ADAM

include("mgd.jl")
include("modified_mgd.jl")
include("spsa.jl")
include("policy_grad.jl")

end # module

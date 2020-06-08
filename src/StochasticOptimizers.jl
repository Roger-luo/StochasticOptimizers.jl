module StochasticOptimizers

using LsqFit
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using Evolutionary
using Evolutionary: AbstractOptimizer
using Distributions
using ProgressMeter
using NiLang: @i, @instr, @assignback
import NiLang
import NiLang: NiLangCore
using NiLang.AD: Grad
using ForwardDiff

import Evolutionary: CMAES, optimize, update_state!, AbstractOptimizerState, minimizer,
    Options, f_calls, trace, value

"""
    optimize(f, x0, method::M)

Optimize function `f` with initial parameter `x0`. `method` can be
* `CMAES` - covariance matrix adaptation evolution strategy
* `SPSA` - simultaneous perturbation stochastic approximation
* `MGD` - model gradient descent

Please type e.g. `?CMAES` in a Julia REPL for help information of each method.
"""
function optimize end

export CMAES, optimize, update_state!, Options, minimizer, f_calls, trace, value
export PolicyGrad, ADAGrad, ADAM

include("mgd.jl")
include("modified_mgd.jl")
include("spsa.jl")
include("policy_grad.jl")

end # module

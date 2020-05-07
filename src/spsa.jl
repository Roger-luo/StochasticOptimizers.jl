export SPSA

"""
    SPSA{ORDER,T,TI} <: AbstractOptimizer

Spall's implementation of SPSA algorithm. `ORDER` can be `1` (first order) or `2` (second order).

The constructor `SPSA{ORDER}(; kwargs...)` takes following keyword arguments:

* `δ` and `ξ` are hyper-parameters that define the search region `δ/m^ξ`, where `m` is the step.
* `γ`, `α` and `A` are hyper-parameters that define the learning rate `γ/(m+A)^α`
* `ϵ` is the convergence tolerence

## Reference
    Spall, J. C. (1998).
    Implementation of the simultaneous perturbation algorithm for stochastic optimization.
    IEEE Transactions on Aerospace and Electronic Systems.
    https://doi.org/10.1109/7.705889
"""
struct SPSA{ORDER,T,BT} <: AbstractOptimizer
    δ::T
    ξ::T
    γ::T
    α::T
    A::T
    bounds::BT
    function SPSA{ORDER}(; δ=0.5, ξ=0.101, γ=0.1, α=0.602, A=10, bounds=nothing) where ORDER
        @instr promote(δ, ξ, γ, α, A)
        T = eltype(δ)
        new{ORDER,T,typeof(bounds)}(δ, ξ, γ, α, A, bounds)
    end
end

"""
    SPSAState{T} <: AbstractOptimizerState

SPSA state. Members are

* `x` is current optimal
* `m` is the iteration number
* `Hk` is the estimated Hessian (only if ORDER == 2)
* `loss_history` is the estimated loss.

Can be constructed as

    SPSAState(x0)

where `x0` is the intial guess.
"""
mutable struct SPSAState{T,YT} <: AbstractOptimizerState
    x::Vector{T}
    m::Int
    Hk::Matrix{T}
    loss_history::Vector{YT}
end

Evolutionary.minimizer(s::SPSAState) = s.x
Evolutionary.value(s::SPSAState) = s.loss_history[end]
Evolutionary.initial_population(method::SPSA, individual::AbstractVector) = [individual]
Evolutionary.initial_state(method::SPSA, option, objfun, population) = SPSAState(first(population), Evolutionary.value(objfun, first(population)))
Evolutionary.default_options(::SPSA) = (iterations=10000, abstol=1e-8)

function SPSAState(x0::AbstractVector{TX}, y0) where TX
    nx = length(x0)
    SPSAState(x0, 0, zeros(TX, nx, nx), [y0])
end

function learning_rate(method::SPSA, k::Integer)
    method.γ / (k + method.A)^method.α
end

function search_range(method::SPSA, k::Integer)
    method.δ / k^method.ξ
end

function Evolutionary.update_state!(objfun, state, population, method::SPSA{1})
    state.m += 1
    k = state.m
    g, delta, fpos, fneg, fexpect = _get_g(objfun, state.x, search_range(method, k))
    state.x .-= learning_rate(method, k) .* g
    push!(state.loss_history, fexpect)
    clip!(state.x, method.bounds)
    false
end

function Evolutionary.update_state!(objfun, state, population, method::SPSA{2})
    state.m += 1
    k = state.m
    ck = search_range(method, k)
    g, delta, pos, neg, fexpect = _get_g(objfun, state.x, ck)
    push!(state.loss_history, fexpect)
    Hk_ = _get_h(objfun, state.x, ck, delta, pos, neg)
    state.Hk .= ((k-1) / k) .* state.Hk .+ (1 / k) .* Hk_
    step = learning_rate(method, k) .* (regularted_inv(state.Hk; delta=1e-5) * g)
    state.x .-= step
    clip!(state.x, method.bounds)
    false
end

clip!(x, ab::Nothing) = x
function clip!(x, ab::Tuple)
    a, b = ab
    @inbounds for i = eachindex(x)
        if x[i] < a
            x[i] = a
        elseif x[i] > b
            x[i] = b
        end
    end
    return x
end

"""
suggest hyper parameters for SPSA.

Args:
    fun (func): loss function.
    x0 (ndarray): initial variables.
    initial_step (float): the initial step size.
    noise_strength (float): the noise level (standard error of fun).

Return:
    (γ, δ): hyper parameters.
"""
function _hp_suggest(fun, x0, initial_step, noise_strength, α, A)
    num_eval = 10  # evaluate gradient 10 times to get the mean step.
    δ = noise_strength * 300 + 1e-4
    g0 = sum(x->_get_g(fun, x0, δ, return_hessian=false), 1:num_eval)/num_eval
    # use g0*ak = initial_step to set parameters
    γ = initial_step / g0 * (1 + A)^α
    return γ, δ
end


"""
compute the gradient
"""
function _get_g(fun, x0, ck)
    p = length(x0)
    delta = rand([-1,1], p) .* ck
    xpos = x0 .+ delta
    xneg = x0 .- delta
    fpos, fneg = Evolutionary.value(fun, xpos), Evolutionary.value(fun,xneg)
    fexpect = (fpos + fneg)/2
    (fpos - fneg) ./ (2 .* delta), delta, (xpos, fpos), (xneg, fneg), fexpect
end

function _get_h(fun, x0, ck, delta, pos, neg)
    xpos, fpos = pos
    xneg, fneg = neg
    delta1 = rand([-1,1], length(x0)) .* ck
    fneg_ = Evolutionary.value(fun, xneg .+ delta1)
    fpos_ = Evolutionary.value(fun, xpos .+ delta1)
    g1n = (fneg_ - fneg) ./ delta1
    g1p = (fpos_ - fpos) ./ delta1
    hessian = (g1p .- g1n) ./ 4. ./ delta'
    hessian .+= hessian'
    return hessian
end

function regularted_inv(A; delta)
    inv(sqrt(A * A) + delta * I)
end

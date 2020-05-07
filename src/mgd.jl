export MGD

# prirating!
function LsqFit.check_data_health(xdata::Vector{<:AbstractVector}, ydata)
    true
end

abstract type AbstractMGD{T} <: AbstractOptimizer end

"""
    MGD{T,TI,BT} <: AbstractMGD{T}

Model gradient descent method, the constructor takes following keyword arguments:

* `δ` and `ξ` are hyper-parameters that define the search region `δ/m^ξ`, where `m` is the step.
* `γ`, `α` and `A` are hyper-parameters that define the learning rate `γ/(m+A)^α`
* `ϵ` is the convergence tolerence
* `k` is the population size
* `n` is the maximum function evaluation
* `bounds` is the parameter upper and lower bounds, default to nothing

For more information about hyper-parameters, see the appendix of: https://arxiv.org/pdf/2004.04197.pdf
"""
struct MGD{T,TI,BT} <: AbstractMGD{T}
    δ::T
    ξ::T
    γ::T
    α::T
    A::T
    k::TI
    n::TI
    bounds::BT
    function MGD(; δ=0.5, ξ=0.101, γ=0.1, α=0.602, A=nothing, k=10, n=10000, bounds=nothing)
        @instr promote(δ, ξ, γ, α)
        @instr promote(k, n)
        T = eltype(δ)
        TI = eltype(k)
        if A === nothing
            A = T(0.1*n/k)
        end
        new{T,TI,typeof(bounds)}(δ, ξ, γ, α, A, k, n, bounds)
    end
end

"""
    MGDState{XT, YT}

Model gradient descent optimizer state. Members are

* `x` is current optimal
* `y` is current optimal function value
* `m` is the iteration number
* `fitted` is the fitted quadrature model
* `fitting_error` is the error in the fitting of current quadrature model

Can be constructed as

    MGDState(x0, y0)

where `x` is the intial guess.
"""
mutable struct MGDState{XT, YT} <: AbstractOptimizerState
    x::XT
    m::Int
    ys::Vector{YT}
    fitted::Vector{Float64}
    fitting_error::Float64
end

function MGDState(x0, y0)
    nx = length(x0)
    MGDState(x0, 0, [y0], zeros(Float64, nx*(nx+1) ÷ 2 + nx + 1), NaN)
end

Evolutionary.minimizer(s::MGDState) = s.x
Evolutionary.value(s::MGDState) = s.ys[end]
Evolutionary.initial_population(method::AbstractMGD, individual::AbstractVector) = [individual]
function Evolutionary.initial_state(method::AbstractMGD, option, objfun, population)
    y = Evolutionary.value(objfun, first(population))
    MGDState(first(population), y)
end
Evolutionary.default_options(::AbstractMGD) = (iterations=10000, abstol=1e-8)

function Evolutionary.update_state!(objfun, state::MGDState{TX,TY}, population::AbstractVector{TX}, method::AbstractMGD) where {TX,TY}
    δ, ξ, γ, k = method.δ, method.ξ, method.γ, method.k
    δi = δ/(state.m+1)^ξ
    for i in 1:k
        x2 = rand_disk(state.x, δi)
        push!(population, x2)
        push!(state.ys, Evolutionary.value(objfun, x2))
    end
    Lx2 = TX[]
    Ly2 = TY[]
    for (x2, y2) in zip(population, state.ys)
        if norm(x2 - state.x) <= δi
            push!(Lx2, x2)
            push!(Ly2, y2)
        end
    end
    fitres = _fit(multivariate_quadratic, Lx2, Ly2, state.fitted)
    state.fitted = fitres.param
    if length(fitres.resid) == 0
        @warn "Quadrature function fitting probably fails, try to use another parameter set for optimization?"
        state.fitting_error = NaN
    else
        state.fitting_error = fitres.resid[end]
    end
    x2 = opt_findnext(state, method)
    state.x = x2
    y = Evolutionary.value(objfun, state.x)
    push!(population, copy(state.x))
    push!(state.ys, y)
    state.m += 1
    Evolutionary.f_calls(objfun) >= method.n
end

function opt_findnext(state, method::MGD)
    g = NiLang.AD.gradient(Val(1), multivariate_quadratic, (0.0, state.x, state.fitted))[2]
    γ2 = method.γ/(state.m+1+method.A)^method.α
    newx = state.x .- γ2 .* g
    clip!(newx, method.bounds)
end

function rand_disk(x, δi)
    while true
        np = rand(length(x)) .* 2δi
        if norm(np) < δi
            return x .+ np
        end
    end
end

function _fit(fit_func, xs, ys, p0)
    function model(xs, p)
        map(x->fit_func(0.0, x, p)[1], xs)
    end
    fit = curve_fit(model, xs, ys, p0)
    fit
end

@i function multivariate_quadratic(res::T, x::AbstractVector, p) where T
    @safe @assert length(x)*(length(x)+1) ÷ 2 + length(x) + 1 == length(p)
    k ← 1
    res += identity(p[k])
    @invcheckoff @inbounds for j = 1:length(x)
        k += identity(1)
        res += x[j] * p[k]
    end
    @invcheckoff @inbounds for j = 1:length(x)
        anc ← zero(T)
        for i=1:j-1
            k += identity(1)
            anc += x[i] * x[j]
            res += anc * p[k]
            anc -= x[i] * x[j]
        end
        k += identity(1)
        anc += x[j] ^ 2
        res +=  anc * p[k]
        anc -= x[j] ^ 2
    end
    k → length(x)*(length(x)+1) ÷ 2 + length(x) + 1
end

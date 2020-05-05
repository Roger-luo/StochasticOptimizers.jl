export MMGD
"""
    MMGD{T,TI,BT} <: AbstractMGD{T}

Modified model gradient descent method, the constructor takes following keyword arguments:

* `δ` and `ξ` are hyper-parameters that define the search region `δ/m^ξ`, where `m` is the step.
* `γ` is the learning rate, default to `0.5`
* `ϵ` is the convergence tolerence
* `k` is the population size
* `n` is the maximum function evaluation
* `bounds` is the parameter upper and lower bounds, default to nothing

For more information about hyper-parameters, see the appendix of: https://arxiv.org/pdf/2004.04197.pdf
"""
struct MMGD{T,TI,BT} <: AbstractMGD{T}
    δ::T
    ξ::T
    γ::T
    ϵ::T
    k::TI
    n::TI
    bounds::BT
    function MMGD(; δ=0.5, ξ=0.101, γ=0.5, ϵ=1e-8, k=10, n=10000, bounds=nothing)
        @instr promote(δ, ξ, γ, ϵ)
        @instr promote(k, n)
        T = eltype(δ)
        TI = eltype(k)
        new{T,TI,typeof(bounds)}(δ, ξ, γ, ϵ, k, n)
    end
end

function opt_findnext(state, method::MMGD)
    x2, ismin = _optimal(multivariate_quadratic, state.fitted)
    newx = ismin ? state.x + (x2-state.x)*method.γ : state.x - (x2-state.x)*method.γ
    clip!(newx, method.bounds)
end

function _optimal(::typeof(multivariate_quadratic), params::AbstractVector{T}) where T
    nx = (isqrt(8*length(params) + 1) - 3) ÷ 2
    b = -params[2:1+nx]
    A = zeros(T, nx, nx)
    k = 1+nx
    for i=1:nx
        for j=1:i-1
            k += 1
            A[i,j] = params[k]
            A[j,i] = params[k]
        end
        k += 1
        A[i,i] = 2*params[k]
    end
    if isposdef(A)
        ismin = true
    else
        ismin = false
    end
    gmres(A, b), ismin
end

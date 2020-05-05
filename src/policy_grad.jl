ignore(f) = f()
Zygote.@adjoint ignore(f) = ignore(f), _->nothing

macro ignore(ex)
    quote
        ignore() do
            $(esc(ex))
        end
    end
end

Zygote.@adjoint function Distributions._normlogpdf(z::T) where T
    z1, back1 = Zygote.pullback(abs2, z)
    z2, back2 = Zygote.pullback(+, z1, T(Distributions.log2π))
    z3, back3 = Zygote.pullback(x->-x/2, z2)

    function normlogpdf_pullback(Δ)
        ∇z2, = back3(Δ)
        ∇z1, _ = back2(∇z2)
        ∇z = back1(∇z1)
        return ∇z
    end
    return z3, normlogpdf_pullback
end

export policy_gradient, ADAM, ADAGrad, update!, expect_loss

function logploss(f, μ::AbstractVector{T}, σ::AbstractVector{T}; nsamples::Int=1000) where T
    ps = Normal.(μ, σ)
    expect_logploss = zero(T)
    for _ in 1:nsamples
        θs = @ignore rand.(ps)
        logp = sum(logpdf.(ps, θs))
        l = @ignore f(θs)
        expect_logploss += l * logp
    end
    return expect_logploss / nsamples
end

function policy_gradient(f, μ::AbstractVector{T}, sqrtσ::AbstractVector{T}; nsamples::Int=1000) where T
    return Zygote.gradient(μ, sqrtσ) do μ, sqrtσ
        logploss(f, μ, sqrtσ.^2; nsamples=nsamples)
    end

    # fd_grad = ForwardDiff.gradient(hcat(μ, sqrtσ)) do x
    #     logploss(f, x[:, 1], x[:, 2].^2; nsamples=nsamples)
    # end
end

function expect_loss(f, μ::AbstractVector{T}, sqrtσ::AbstractVector{T}; nsamples::Int=1000) where T
    ps = Normal.(μ, sqrtσ.^2)
    return mean(f(rand.(ps)) for _ in 1:nsamples)
end

const ϵ = 1e-8
## Copied from Flux
mutable struct ADAM
    eta::Float64
    beta::Tuple{Float64,Float64}
    state::IdDict
end

ADAM(η = 0.001, β = (0.9, 0.999)) = ADAM(η, β, IdDict())

function apply!(o::ADAM, x, Δ)
    η, β = o.eta, o.beta
    mt, vt, βp = get!(o.state, x, (zero(x), zero(x), β))
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ^2
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
    o.state[x] = (mt, vt, βp .* β)
    return Δ
end

mutable struct ADAGrad
    eta::Float64
    acc::IdDict
end

ADAGrad(η = 0.1) = ADAGrad(η, IdDict())

function apply!(o::ADAGrad, x, Δ)
  η = o.eta
  acc = get!(o.acc, x, fill!(zero(x), ϵ))::typeof(x)
  @. acc += Δ^2
  @. Δ *= η / (√acc + ϵ)
end

function update!(opt, x::AbstractArray, x̄)
    x .-= apply!(opt, x, x̄)
    return
end

@generated function update!(opt, m::M, ∇m) where M
    body = Expr(:block)
    for each in fieldnames(M)
        each = QuoteNode(each)
        push!(body.args, :(update!(opt, getfield(m, $each), getfield(∇m, $each))))
    end
    return body
end

update!(x::AbstractArray, x̄::Nothing) = nothing
update!(opt, x::AbstractArray, x̄::Nothing) = nothing
update!(opt, m::M, ∇m::Nothing) where M = nothing

"""
    PolicyGrad(;grad_optimizer=ADAGrad(), maxiter=2000, nsamples=100, showprogress=true)

Policy gradient optimizer. The ansatz is a hard coded normal distribution for now.

# Example

```julia
μ, sqrtσ = ones(2) + 1e-2 * randn(2), ones(2)
μ, sqrtσ = optimize(noisy_rosenbrock, μ, sqrtσ , PolicyGrad(showprogress=false))
expect = mean(noisy_rosenbrock(rand.(Normal.(μ, sqrtσ.^2))) for _ in 1:1000)
```
"""
Base.@kwdef struct PolicyGrad <: AbstractOptimizer
    grad_optimizer=ADAGrad()
    maxiter::Int=2000
    nsamples::Int=100
    showprogress::Bool=true
end

function Evolutionary.optimize(f, μ0::AbstractVector, sqrtσ0::AbstractVector, opt::PolicyGrad)
    μ, sqrtσ = copy(μ0), copy(sqrtσ0)

    if opt.showprogress
        @showprogress for _ in 1:opt.maxiter
            gs = policy_gradient(f, μ, sqrtσ; nsamples=opt.nsamples)
            update!(opt.grad_optimizer, (μ, sqrtσ), gs)
        end
    else
        for _ in 1:opt.maxiter
            gs = policy_gradient(f, μ, sqrtσ; nsamples=opt.nsamples)
            update!(opt.grad_optimizer, (μ, sqrtσ), gs)
        end
    end

    return μ, sqrtσ
end

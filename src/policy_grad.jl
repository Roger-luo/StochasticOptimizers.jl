using Distributions
using Zygote

ignore(f) = f()
Zygote.@adjoint ignore(f) = ignore(f), _->nothing

macro ignore(ex)
    quote
        ignore() do
            $(esc(ex))
        end
    end
end

Zygote.@adjoint function Distributions.normlogpdf(z::Number)
    z1, back1 = pullback(abs2, z)
    z2, back2 = pullback(x->-x/2, z1 + Distributions.log2π)
    function normlogpdf_pullback(Δ)
        ∇z1, = back2(Δ)
        ∇z = back1(∇z1)
        return ∇z
    end
    return z2, normlogpdf_pullback
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

Base.@kwdef struct PolicyGrad <: AbstractOptimizer
    grad_optimizer=ADAGrad()
    maxiter::Int=2000
    nsamples::Int=100
end

function Evolutionary.optimize(f, μ0::AbstractVector, sqrtσ0::AbstractVector, opt::PolicyGrad)
    μ, sqrtσ = copy(μ0), copy(sqrtσ0)
    for _ in 1:opt.maxiter
        gs = policy_gradient(f, μ, sqrtσ; nsamples=opt.nsamples)
        update!(opt.grad_optimizer, (μ, sqrtσ), gs)
    end

    d = Normal.(μ, sqrtσ)
    return rand.(d)
end

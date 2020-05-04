using StochasticOptimizers
using Distributions
using Test, Random
Random.seed!(42)
function noisy_rosenbrock(x::Vector)
    x = x + 1e-1 * randn(length(x))
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

@testset "policy grad"
    μ, sqrtσ = ones(2) + 1e-2 * randn(2), ones(2)
    # mean(noisy_rosenbrock(rand.(Normal.(μ, sqrtσ.^2))) for _ in 1:1000)

    μ, sqrtσ = optimize(noisy_rosenbrock, μ, sqrtσ , PolicyGrad(showprogress=false))
    expect = mean(noisy_rosenbrock(rand.(Normal.(μ, sqrtσ.^2))) for _ in 1:1000)

    @test expect < 3
end

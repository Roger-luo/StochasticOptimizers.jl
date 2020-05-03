using LinearAlgebra
using Test, Random
using StochasticOptimizers

@testset "CMA-ES RosenBrock" begin
	Random.seed!(2)
	N = 2
    rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
	# termination condition
	result = optimize(rosenbrock, randn(2), CMAES(;μ=5, λ=100))
	@test ≈(result.minimizer, ones(N), atol=1e-2)
    @test ≈(result.minimum, 0.0, atol=1e-2)
end

using Random
Random.seed!(2)
N = 2
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
using Evolutionary
# termination condition
using BenchmarkTools
@benchmark result = Evolutionary.optimize(rosenbrock, randn(2), $(CMAES(;μ=5, λ=100)))

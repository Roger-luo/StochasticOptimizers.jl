using LinearAlgebra
using Test, Random
using StochasticOptimizers

@testset "CMA-ES RosenBrock" begin
	Random.seed!(2)
	N = 2
    rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
	# termination condition
	result = optimize(rosenbrock, randn(2), CMAES(;μ=5, λ=100))
	@show result
	@test ≈(result.minimizer, ones(N), atol=1e-2)
    @test ≈(result.minimum, 0.0, atol=1e-2)
end

@testset "CMA-ES RosenBrock" begin
	Random.seed!(2)
	N = 2
    rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
	# termination condition
	result0 = optimize(rosenbrock, [5.5, -1.1], CMAES(;μ=5, λ=100, σ0=1.5))
	result = optimize(rosenbrock, [5.5, -1.1], CMAES(;μ=5, λ=100, σ0=0.1))
	@test ≈(result.minimizer, ones(N), atol=1e-2)
    @test ≈(result.minimum, 0.0, atol=1e-2)
	@test result.iterations > result0.iterations
end

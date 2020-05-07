using Test, Random
using StochasticOptimizers
using Evolutionary

@testset "SPSA" begin
    rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
	Random.seed!(2)
	x0 = randn(2)
    opt = SPSA{1}(bounds=(-1, 2),
        γ=0.2, δ=0.1)
	# to trace results
	# optimize(rosenbrock, randn(2), opt, )
	res = optimize(rosenbrock, randn(2), opt, Options(store_trace=true, iterations=20000))
	@test ≈(res.minimizer, ones(2), atol=1e-2)
    @test ≈(minimum(res), 0.0, atol=1e-3)
end

@testset "SPSA 2nd order" begin
    rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
	Random.seed!(3)
	x0 = randn(2)
	# second order function
    opt = SPSA{2}(bounds=(-1, 2),
        γ=10.0, δ=0.03)
	res = optimize(rosenbrock, randn(2), opt, Options(iterations=10000))
	@test ≈(res.minimizer, ones(2), atol=1e-2)
    @test ≈(res.minimum, 0.0, atol=1e-3)
end

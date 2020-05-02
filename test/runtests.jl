using StochasticOptimizers
using Test

@testset "cmaes" begin
    include("cmaes.jl")
end

@testset "mgd" begin
    include("mgd.jl")
end

@testset "modified mgd" begin
    include("modified_mgd.jl")
end

@testset "spsa" begin
    include("spsa.jl")
end

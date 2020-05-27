include("client.jl")

using Test
@testset "postfunc" begin
    spins = [0 1; 1 1; 1 1]
    @test postfunc(spins) == 5/3
end

@testset "emulator backend" begin
    res = run_experiment(EmulatorConfig(rand_unit_disk_graph(10, 1.5), 200), rand(2,2))
    @test size(res) == (200, 10)
end

include("client.jl")

# configure the loss function
function postfunc(spinconfigs)
    -sum(spinconfigs)/size(spinconfigs,1)
end
loss = cloud_f(postfunc; params_ncol=2, port=2021,
    overwrite_file=true, output_folder=joinpath(@__DIR__, "data"))

# optimize with first order SPSA
#opt = SPSA{1}(bounds=(-1, 2),γ=0.2, δ=0.1)
opt = CMAES(μ=8, λ=40)

using Random
Random.seed!(2)
res = optroutine(loss, rand(16), opt; max_call=10000)

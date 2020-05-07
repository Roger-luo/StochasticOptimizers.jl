using StochasticOptimizers
using StochasticOptimizers: value

function optroutine(f, x0::AbstractVector{TX}, opt::SPSA; max_call) where {TX}
    # define the objective and state
    objfun = NonDifferentiable(f, x0)
    state = StochasticOptimizers.SPSAState(x0, value(objfun, x0))
    for i = 1:10000
        # convergence condition
        f_calls(objfun) > max_call && break
        x_pre = state.x

        # update state
        update_state!(objfun, state, nothing, opt)

        # print infomations
        i % 100 == 1 && println("""
# Summary
    iteration: $(state.m),
    nfeval: $(f_calls(objfun)),
    optimal x: $(state.x),
    optimal f: $(value(state)),
""")
    end
end

rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

# optimize with first order SPSA
opt = SPSA{1}(bounds=(-1, 2),γ=0.2, δ=0.1)
res = optroutine(rosenbrock, randn(2), opt; max_call=10000)

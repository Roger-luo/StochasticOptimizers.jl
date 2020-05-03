using StochasticOptimizers

function optroutine(objfun, x0::AbstractVector{TX}, opt::SPSA) where {TX}
    # define a state
    state = StochasticOptimizers.SPSAState(x0)
    for i = 1:opt.n
        # convergence condition
        norm(state.step) < opt.ϵ && break

        # update state
        update_state!(objfun, state, nothing, opt)

        # print infomations
        i % 100 == 1 && println("""
# Summary
    iteration: $(state.m),
    nfeval: $(state.neval),
    optimal x: $(state.x),
    optimal f: $(objfun(state.x)),
    current step: $(state.step).
""")
    end
end

rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

# optimize with first order SPSA
opt = SPSA{1}(bounds=(-1, 2),γ=0.2, δ=0.1, n=20000, ϵ=1e-10)
res = optroutine(rosenbrock, randn(2), opt)

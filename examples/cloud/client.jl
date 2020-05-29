using DelimitedFiles
using Sockets
using StochasticOptimizers, Evolutionary
using StochasticOptimizers: value, NonDifferentiable

"""
    cloud_f(postfunc; params_ncol, port)

* `postfunc` isa a function that take spin configurations as input, and the loss as output.
* `params_ncol` is the number of data columns.
* `port` is the socket application port.
"""
function cloud_f(postfunc; params_ncol::Int, port::Int,
        overwrite_file::Bool=false, output_folder=@__DIR__)
    k = 0
    function loss(x)
        c = connect(port)
        if !isopen(c)
            @debug "connection fail!"
        else
            @debug "connection success!"
        end

        # send file
        # you might want to rescale your data here
        k += 1
        ifname = joinpath(output_folder, "Parameters_$k.dat")
        if isfile(ifname) && !overwrite_file
            error("file name conflict! got input file $ifname.")
        end
        writedlm(ifname, reshape(x,:,params_ncol))
        println(c, ifname)

        # receive file
        local ofname
        while isopen(c)
            ofname = readline(c, keep=false)
            ofname
        end
        close(c)
        @debug "Get datafile" ofname
        if !isfile(ofname)
            error("output file not found! got output file $ofname.")
        end
        spinconfigs = readdlm(ofname, Int)
        postfunc(spinconfigs)
    end
end

function optroutine(f, x0::AbstractVector{TX}, opt; max_call, print_step=10) where {TX}
    # define the objective and state
    objfun = NonDifferentiable(f, x0)
    state = StochasticOptimizers.SPSAState(x0, value(objfun, x0))
    population = Evolutionary.initial_population(opt, x0)
    state = Evolutionary.initial_state(opt, Evolutionary.Options(), objfun, population)
    for i = 1:10000
        # convergence condition
        f_calls(objfun) > max_call && break

        # update state
        update_state!(objfun, state, population, opt)

        # print infomations
        i % print_step == 1 && println("""
# Summary
    iteration: $(i),
    nfeval: $(f_calls(objfun)),
    optimal f: $(value(state)),
""")
    end
end

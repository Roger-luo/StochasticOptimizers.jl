using Sockets
using DelimitedFiles

abstract type ExperimentConfig end

function run_server(config; port::Int, output_folder=@__DIR__)
    k = Ref(1)
    # a simple server (device side)
    t = @async begin
        server = listen(port)
        while true
            sock = accept(server)
            @async while isopen(sock)
                s = readline(sock, keep=false)
                if isempty(s)
                    continue
                end
                @debug "Get file" s
                if !isfile(s)
                    @debug @debug "Not a file" s
                    write(sock, "not a file!")
                    continue
                end
                params = readdlm(s)
                @debug "Get params" params
                # 200 measurements
                result = run_experiment(config, params)
                ofname = joinpath(output_folder, "Measurementdata_$(k[]).dat")
                writedlm(ofname, result)
                write(sock, ofname)
                @debug "Data sent!"
                close(sock)
                k[] += 1
            end
        end
    end

    wait(t)
end

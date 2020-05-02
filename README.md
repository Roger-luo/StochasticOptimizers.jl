# StochasticOptimizers

Optimizers for noisy loss functions.

[![Build Status](https://travis-ci.com/Happy-Diode/StochasticOptimizers.jl.svg?branch=master)](https://travis-ci.com/Happy-Diode/StochasticOptimizers.jl)
[![Codecov](https://codecov.io/gh/Happy-Diode/StochasticOptimizers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Happy-Diode/StochasticOptimizers.jl)


## Features

* [x] SPSA
* [x] CMAES (from Evolutionary)
* [x] ModelGradientDescent
* [x] Policy Gradient
* [ ] Post an issue to motivate us add more, please include the paper with the algorithm.

## To start

To install, open a Julia REPL, type `]` to enter `pkg` mode, then
```julia pkg
pkg> add https://github.com/Happy-Diode/StochasticOptimizers.jl
```

To use, open a Julia REPL and type
```julia
julia> optimizer = CMAES()

julia> opt_result = optimize(f, x0, optimizer)
```

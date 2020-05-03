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

Our interface is compatible with [Evolutionary.jl](https://github.com/wildart/Evolutionary.jl).
To start using, open a Julia REPL and type

```julia
using StochasticOptimizers

# define a loss
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

# optimize with first order SPSA
opt = SPSA{1}(bounds=(-1, 2),γ=0.2, δ=0.1, n=20000, ϵ=1e-10)
res = optimize(rosenbrock, randn(2), opt)
```

Similarly, you can change optimizers to

* CMA-ES
```julia
opt = CMAES(;μ=5, λ=100)
```
* Model Gradient Descent
```julia
opt = MGD(γ=0.5, δ=0.6, k=10, A=2.0, ϵ=1e-8, n=10000)
```

and other methods defined in [Evolutionary.jl](https://github.com/wildart/Evolutionary.jl).

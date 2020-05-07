# StochasticOptimizers

Optimizers for noisy loss functions.

[![Build Status](https://travis-ci.com/GiggleLiu/StochasticOptimizers.jl.svg?branch=master)](https://travis-ci.com/GiggleLiu/StochasticOptimizers.jl)
[![Codecov](https://codecov.io/gh/GiggleLiu/StochasticOptimizers.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GiggleLiu/StochasticOptimizers.jl)


## Features

* [x] SPSA
* [x] CMAES (from Evolutionary)
* [x] ModelGradientDescent
* [x] Policy Gradient
* [ ] Post an issue to motivate us add more, please include the paper with the algorithm.

## To start

To install, open a Julia REPL, type `]` to enter `pkg` mode, then
```julia pkg
pkg> add https://github.com/GiggleLiu/StochasticOptimizers.jl
```

Our interface is compatible with [Evolutionary.jl](https://github.com/wildart/Evolutionary.jl).
To start using, open a Julia REPL and type

```julia
using StochasticOptimizers

# define a loss
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

# optimize with first order SPSA
opt = SPSA{1}(bounds=(-1, 2),γ=0.2, δ=0.1)
res = optimize(rosenbrock, randn(2), opt, Options(iterations=20000, store_trace=true))
```

Similarly, you can change optimizers to

* CMA-ES
```julia
opt = CMAES(;μ=5, λ=100)
```
* Model Gradient Descent
```julia
opt = MGD(γ=0.5, δ=0.6, k=10, A=2.0, n=10000)
```

and other methods defined in [Evolutionary.jl](https://github.com/wildart/Evolutionary.jl).

For step-wise update, see `examples/stepwise.jl`.

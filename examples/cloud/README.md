# Cloud Optimizer

### The server side
Run the Rydberg Emulator backend, data are transported as files.

```bash
$ julia emulated_server.jl
```

Note: Should install `EliminateGraphs` for computing the exact solution.
install `RydbergEmulator` for simulator backend. Just type

```julia pkg
pkg> add https://github.com/GiggleLiu/EliminateGraphs.jl.git

pkg> add https://github.com/Happy-Diode/RydbergEmulator.jl.git
```

### The client side

Run the the client side, using CMA-ES optimizer.

```bash
$ julia rydberg_client.jl
```

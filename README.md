# ParametricMachinesDemos

[![CI](https://github.com/BeaverResearch/ParametricMachinesDemos.jl/workflows/CI/badge.svg?branch=main)](https://github.com/BeaverResearch/ParametricMachinesDemos.jl/actions?query=workflow%3ACI+branch%3Amain)
[![codecov.io](http://codecov.io/github/BeaverResearch/ParametricMachinesDemos.jl/coverage.svg?branch=main)](http://codecov.io/github/BeaverResearch/ParametricMachinesDemos.jl?branch=main)

This repository contains some examples of _parametric machines_:

- `DenseMachine`,
- `ConvMachine`,
- `RecurMachine`.

## Reproducing figures

To reproduce figures (relative to the parametric machines manuscript), clone the repository.
Then, from the top-level, initialize Julia (version 1.7.2 was used) with the `figures` project:

```
julia --project=figures
```

Use `Pkg.instantiate` to install all required dependencies.

```julia-repl
julia> import Pkg; Pkg.instantiate()
```

Run

```julia
julia> include("figures/run_benchmarks.jl")
```

to generate the benchmark results (note that this requires CUDA).

Finally, run

```julia
julia> include("figures/benchmarks.jl")
```

and

```julia
julia> include("figures/depthsequence.jl")
```

to generate the figures.
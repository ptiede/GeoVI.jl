# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run tests interactively (from Julia REPL)
julia --project
julia> include("test/runtests.jl")
```

### Formatting
The project uses [Runic.jl](https://github.com/fredrikekre/Runic.jl) for code formatting:
```bash
julia --project -e 'using Runic; Runic.format_file("src/GeoVI.jl")'
```

## Architecture

GeoVI implements variational inference using Fisher-metric geometry. The key assumption is **white-latent**: all latent parameters are a single dense array `ξ` with IID standard normal prior. The posterior objective is `-logdensity(likelihood, forward(ξ)) + 0.5 * dot(ξ, ξ)`.

### Layers (bottom to top)

1. **Likelihoods** (`src/likelihoods.jl`, `src/likelihoods/`) — Define observation models with Fisher geometry. Abstract type `AbstractLikelihood` requires `logdensity`, `normalized_residual`, and `transformation`/`leftsqrtmetric`/`rightsqrtmetric`. `ComposedLikelihood` pulls back a base likelihood through a forward map.

2. **Sampling** (`src/sampling.jl`) — Draws residuals from the posterior metric `I + Fisher` via conjugate gradient. Returns `LinearResidualDraw` with CG convergence info.

3. **Nonlinear update** (`src/nonlinear.jl`) — Refines linear residuals via Newton-CG or gradient-based optimizers. `NewtonCG` is the recommended optimizer for the inner loop.

4. **Outer VI loop** (`src/vi.jl`) — Coordinates sampling and position optimization. `fit(problem)` runs the full loop; `step_vi(problem, samples, state)` runs one iteration. `VIConfig` holds all hyperparameters.

5. **Reactant extension** (`ext/GeoVIReactantExt.jl`) — Compiles the VI step via Reactant for GPU/TPU. Loaded automatically when `Reactant` is available; uses `AutoReactant` AD backend. Caches compiled steps in `ReactantVIStepCache`.

### Key types

- `Samples` — holds `position` (expansion point) and `residuals` (relative to position); `posterior_samples(s)` returns `position .+ residuals`
- `VIConfig` / `VIState` / `VariationalProblem` — immutable configuration and mutable state for the VI loop
- `MGVIFamily` / `GeoVIFamily` — selects linear-only vs. linear+nonlinear residual updates
- `ReverseKL` — currently the only implemented divergence
- `OptimizationResult` — returned by inner and outer optimizers with convergence info

### Extension points (from `docs/src/interfaces.md`)

- Custom likelihoods: subtype `AbstractLikelihood`, implement required methods
- Custom variational families: subtype `AbstractVariationalFamily`, implement `_draw_sample_block`
- Custom divergences: subtype `AbstractFDivergence`, implement `_fdivergence_value` and `_fdivergence_fishermetric`
- Custom optimizers: subtype `AbstractOptimizer`, implement `_optimize` (and optionally `_optimizer_state`)
- Custom AD backends: implement `_value_and_gradient`, `_automatic_linearize` for your `ADTypes` backend

### Typical usage pattern

```julia
lh = compose(GaussianLikelihood(data; precision=inv_cov), forward_model)
problem = VariationalProblem(lh, xi0; family=GeoVIFamily(), config=VIConfig(...))
samples, state = fit(problem; rng=MersenneTwister(42))
draws = posterior_samples(samples)
```

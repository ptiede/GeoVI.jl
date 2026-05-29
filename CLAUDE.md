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

The full suite lives in a single `test/runtests.jl` with nested `@testset` blocks. To run only one group, comment out siblings or wrap the desired block in its own `@testset` and `include` it from the REPL — there is no per-file test runner.

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

4. **Outer VI loop** (`src/vi.jl`) — Coordinates sampling and position optimization, organized as four orthogonal axes (see "Key types"). The primary interface is the in-place loop: `rng, state = init(rng, problem)` then `step_vi!(rng, problem, state)` per iteration, which mutates the one `VIState` and its buffers in place (the user owns the loop, Optimisers.jl-style; `problem` and `rng` are passed in, not stored in the state). `fit(problem, n; rng)` is a convenience that runs `n` iterations and returns a `VariationalPosterior`. Under Reactant the in-place step `_vi_step!` is compiled once at `init` (Reactant traces the mutation directly). A VI step is the **reparameterization structure of VI sliced into three phases** (`src/vi.jl`, unexported primitives the user can compose): `GeoVI.sample!` (draw white noise `ξ_w ∼ N(0,I)` — the only stochastic, `rng`-using phase), `GeoVI.transform!` (de-whiten: MGVI CG solve, geoVI CG + curve, at the current mean — re-running it with the stored noise recomputes the Fisher at a moved mean), `GeoVI.update!` (estimate the KL and move the variational mean). `step_vi!` = `sample!`→`transform!`→`update!`; custom loops (e.g. draw once, then refine the mean against a fixed realization) call the phases directly. `VIState` stores the per-base-draw white noise (`metric_white`/`prior_white`) so `transform!` can replay a draw at a new mean without RNG state.

5. **AD/compilation extensions** (`ext/`) —
   - `GeoVIEnzymeExt.jl` provides `pushforward`/`_value_and_gradient` via Enzyme (`AutoEnzyme`).
   - `GeoVIReactantExt.jl` compiles the VI step via Reactant for GPU/TPU; uses `AutoReactant` and caches compiled steps in `ReactantVIStepCache`.
   Both load automatically when their package is available.

### Reactant tracing convention (preserve when editing)

Hot loops use `ReactantCore.@trace` so they run under both eager and traced execution. This spans `src/cg.jl` (inner CG), `src/optimize.jl` (Newton/line-search loops), `src/vi.jl` (the n-sample draw loop and Monte Carlo divergence/metric sums), and `ext/GeoVIReactantExt.jl`. Conventions to keep:

- Loops over heavyweight closures use `@trace track_numbers = false` so the body compiles to a single MLIR while/for body instead of `n` trace-time-unrolled copies. Unrolling each iteration's full forward model (e.g. a VLBI NUFT registers O(10) broadcast functions) compounds against Reactant's per-name uniquing cap of 10000.
- `track_numbers = false` keeps the type-walk away from plain `Int`/`Bool` fields in closure environments — so accumulators must already be traced (e.g. seed with `zero(eltype(position))`, not a literal `0`).
- A loop index `i` arrives as a `Reactant.TracedRNumber{<:Integer}` inside `@trace for`; index residual blocks through the `_sample_slice`/`_sample_position` helpers rather than plain getindex. See inline comments in `src/vi.jl` and `src/samples.jl`.
- A new loop-carried scalar must be a *distinct* `TracedRNumber`, not an alias of another carried value. E.g. `prev_value = value` (both already traced) makes two carry slots share one node and breaks `@trace while`'s arg/result matching; init it as `value + zero(value)` to force a fresh node. See `_optimize(::NewtonCG)` in `src/optimize.jl` (`prev_value`, the lagged energy for the CG `absdelta` coupling).
- Runtime branches on a *traced* value must use `@trace if` (or `ifelse`), never a plain `if`. A plain `if` is fine only when its condition is compile-time known — including type-level checks like `absdelta === nothing` (which stays plain on purpose: it gates whether the energy-criterion code is emitted at all, while the value comparison `energy_diff < absdelta` goes through `@trace if`). See `_cg_iterate` in `src/cg.jl`.
- Do NOT nest `@trace if` inside another `@trace if`, and do NOT reference a qualified module name (e.g. `ReactantCore.foo`) inside a `@trace if` body — the former fails macro expansion, the latter resolves in a local scope (`UndefVarError`). Hoist such checks above the `@trace if` into a plain `Bool`.

### The four axes (organizing principle)

A VI step is: **(1)** draw an MC sample set from `q` at the current mean, **(2)** move the mean to minimize the MC-estimated divergence. The knobs separate into four orthogonal axes passed directly to `VariationalProblem` (there is no `VIConfig`):

- **family** (`MGVIFamily` / `GeoVIFamily`) — *how to draw one sample from `q`*, plus the solvers that draw needs. `MGVIFamily(; solver=ConjugateGradient(...))` does the linear metric draw; `GeoVIFamily(; solver=ConjugateGradient(...), curve=NewtonCG(...))` adds the nonlinear curve. The family carries no outer-fit knobs.
- **estimator** (`MCEstimator(; n_samples, mirrored)`) — *how many MC nodes* estimate `E_q[·]`. Family-agnostic; the curve/CG tolerances are NOT here.
- **optimizer** (`NewtonCG(...)` or a bare `Optimisers.jl` rule) — the *outer position update only*. `NewtonCG` is a to-convergence solver carrying flat CG keywords (`cg_rtol`, …); a bare `Optimisers.jl` rule (e.g. `Optimisers.Adam(0.05)`) takes one gradient step per `step_vi!`, with the user's loop providing the iteration count. `NewtonCG` energy-based convergence is opt-in via `absdelta` (absolute) or `delta` (per-d.o.f.; `absdelta = delta·length(x0)` at solve time, cf. NIFTy.re) — setting either also enables the inner-CG energy-decrease coupling (`cg_absdelta = absdelta/100` first iter, `0.1·(last Newton gain)` after). Default `NewtonCG()` is uncoupled.
- **divergence** (`ReverseKL`; `ForwardKL` is a stub) — the objective form. `_fdivergence_value`/`_fdivergence_fishermetric` dispatch jointly on `(family, divergence)`.

### Key types

- `Samples` — holds `position` (expansion point) and `residuals` (relative to position); `posterior_samples(s)` returns `position .+ residuals`
- `VariationalProblem` — bundles likelihood + the four axes + adtype (immutable)
- `VIState` — the mutable *numeric* loop state allocated once by `init` (position, residual buffer, per-base-draw white noise for sample reuse, optimizer state, compile cache); advanced in place by `step_vi!`. The problem and rng are kept *out* of it and passed to `step_vi!` separately
- `VariationalPosterior` — the fitted distribution; supports `rand(rng, post[, n])` and `mean(post)`, and retains the fitting `samples`
- `ConjugateGradient` — the linear solver; user-facing as a family `solver`, internal inside `NewtonCG`
- `OptimizationResult` — returned by inner and outer optimizers with convergence info

### Extension points (from `docs/src/interfaces.md`)

- Custom likelihoods: subtype `AbstractLikelihood`, implement required methods
- Custom variational families: subtype `AbstractVariationalFamily`, implement `_draw_sample_block` and `_draw_one_residual` (consume the family's own solvers)
- Custom divergences: subtype `AbstractFDivergence`, implement `_fdivergence_value(family, divergence, …)` and `_fdivergence_fishermetric(family, divergence, …)` — joint `family × divergence` dispatch
- Custom estimators: subtype `AbstractEstimator`
- Custom optimizers: subtype `AbstractOptimizer`, implement `_optimize` (and optionally `_optimizer_state`)
- Custom AD backends: implement `_value_and_gradient`, `_automatic_linearize` for your `ADTypes` backend

### Typical usage pattern

```julia
lh = compose(GaussianLikelihood(data; precision=inv_cov), forward_model)
problem = VariationalProblem(lh, xi0;
    family    = GeoVIFamily(; solver=ConjugateGradient(rtol=1e-6), curve=NewtonCG(maxiter=10)),
    divergence= ReverseKL(),
    estimator = MCEstimator(; n_samples=8, mirrored=true),
    optimizer = NewtonCG(; maxiter=20, xtol=1e-5, cg_rtol=1e-8),
)

# explicit loop (primary) — rng and problem are passed, not stored in state:
rng, state = init(MersenneTwister(42), problem)
for _ in 1:8
    step_vi!(rng, problem, state)
end
post = posterior(problem, state)

# or the convenience driver:
post = fit(problem, 8; rng=MersenneTwister(42))

draws = rand(MersenneTwister(0), post, 100)   # draw arbitrarily many new samples
μ     = mean(post)                            # the latent mean
```

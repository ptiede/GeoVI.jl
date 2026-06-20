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

**One KL, sliced into a frozen prefix and a differentiated suffix.** VI minimizes the reverse KL `J(θ) = mean_i[-log p(ξ_i) + log q_θ(ξ_i)]` with `ξ_i = T_θ(η_i)`, `η_i ∼ N(0,I)`. A family is the distribution `q_θ`; the *only* axis families differ on is **where the reparameterization `T_θ` is cut** into a part *frozen* at the current `θ` and a part *differentiated* through `θ`:
- `draw_samples!(family, lh, θ, residuals, rng, mirrored)` — the **frozen prefix**: fill the residual buffer with the Monte-Carlo set drawn at the current `θ` and held constant during the update. IID white noise for pushforward families; a CG solve (MGVI) or CG + nonlinear curve (geoVI) for the Fisher-Gaussian families. Expensive, not differentiated, redrawn each outer step. The family owns its rng use and loop. The built-ins bulk-draw the white noise before the `@trace` loop (the JAX/NIFTy split-keys-up-front idiom), but that is a style, not a rule — a traced rng (`ReactantRNG`) may be used inside `@trace` loops when needed.
- `transport_and_logjac(family, θ, r)` — the **differentiated suffix**: returns `(ξ, logjac)`, reconstructing `ξ` from `θ` + the stored residual *and* the reparameterization's log-Jacobian `log|det J|` (the variational entropy term). Default `(θ .+ r, 0)` (MGVI/geoVI: `θ` *is* the latent point, metric log-det dropped); mean-field: `(μ + σ⊙ε, Σlogσ)`; a flow: `(T_θ(ε), log|det J_θ|)` — one forward pass yields both.

Everything else *follows* from the cut: the reverse-KL objective is `mean_i[-log p(ξ_i) - logjac_i]`. Freezing the covariance ⇒ `logjac = 0` (MGVI/geoVI's fixed-metric approximation, the intractable metric log-det dropped); differentiating the shape ⇒ `logjac` carries `Σlogσ` (mean-field) / `log|det J|` (flow). The natural-gradient metric (`natural_gradient_metric`) is the frozen covariance `(I+Fisher)` reused as the `NewtonCG` preconditioner — an *optimizer* concern (consulted only by `NewtonCG`, gated by `supports_natural_gradient`), not a divergence one.

**Parameter container vs latent point.** The *variational parameters* `θ` (what the optimizer moves; `VIState.position`) are decoupled from the *latent point* `ξ` (what the likelihood consumes — a flat white array). For MGVI/geoVI they coincide (`θ` *is* the latent mean, so the `transport_and_logjac` default applies and no override is needed). Mean-field's `θ = (; mean, logstd)` is a NamedTuple, so it overrides `init_params`/`transport_and_logjac` (the σ-scaling lives in `transport_and_logjac` to stay differentiable). `θ` is a Functors-compatible container, so `Optimisers.setup`/`update` and Enzyme/Reactant AD handle it natively; latent-space machinery (the residual buffer, `_sample_slice`, the `0.5·dot(ξ,ξ)` prior, the `@trace` MC loops) stays array-based. Tree-generic helpers (`_param_sub`/`_param_norm`/`_param_all_finite` in `src/tree_utils.jl`, `fmap(copyto!, …)` in `update!`, `Optimisers.destructure` for AutoFiniteDiff on a NamedTuple) cover the `θ`-space ops. There is no `mean` / `latent_mean` / `latent_template`; the fitted output is a *pure* distribution (`rand`, with no retained samples — draw fresh ones), and all buffers are sized from the user's initial `ξ₀`.

### Layers (bottom to top)

1. **Likelihoods** (`src/likelihoods.jl`, `src/likelihoods/`) — Define observation models with Fisher geometry. Abstract type `AbstractLikelihood` requires `logdensity`, `normalized_residual`, and `transformation`/`leftsqrtmetric`/`rightsqrtmetric`. `ComposedLikelihood` pulls back a base likelihood through a forward map.

2. **Sampling** (`src/sampling.jl`) — Draws residuals from the posterior metric `I + Fisher` via conjugate gradient. Returns `LinearResidualDraw` with CG convergence info.

3. **Nonlinear update** (`src/nonlinear.jl`) — Refines linear residuals via Newton-CG or gradient-based optimizers. `NewtonCG` is the recommended optimizer for the inner loop.

4. **Outer VI loop** (`src/vi.jl`) — Coordinates sampling and position optimization, organized as four orthogonal axes (see "Key types"). The primary interface is the in-place loop: `rng, state = init(rng, problem)` then `step_vi!(rng, problem, state)` per iteration, which mutates the one `VIState` and its buffers in place (the user owns the loop, Optimisers.jl-style; `problem` and `rng` are passed in, not stored in the state). `fit([rng], problem, n)` is a convenience that runs `n` iterations and returns the fitted variational distribution (`distribution(problem, state)`, an `AbstractVariationalDistribution`) (rng positional or auto). Under Reactant `step_vi!` is a pure in-place mutation with no host-only state, so the user `@compile`s it themselves and loops the compiled thunk (`fit` does this for you: it compiles `step_vi!` once via `_run_vi!`, then loops the thunk on the host). A VI step is the **`draw → optimize-against-fixed-samples → resample` loop** (the same one NIFTy's `OptimizeVI.update` runs), sliced into **two phases** (`src/vi.jl`, unexported primitives the user can compose): `GeoVI.draw_samples!` (the only stochastic, `rng`-using phase — fill `state.residuals` with the MC set drawn at the current mean: pushforward = IID white noise, MGVI = CG solve, geoVI = CG + curve), then `GeoVI.update!` (estimate the KL with that fixed set and move the variational parameters — one `Optimisers` step, or `NewtonCG` to convergence). `step_vi!` = `draw_samples!`→`update!`; a custom loop calls the two phases directly. The white noise the draw consumes is transient scratch allocated inside `draw_samples!` (the built-ins bulk-draw it before their `@trace` loop, though rng use inside a traced loop is also fine), not stored in `VIState` — the state holds only `position`, the latent-shaped `residuals` buffer, and `optimizer_state`.

5. **AD/compilation extensions** (`ext/`) —
   - `GeoVIEnzymeExt.jl` provides `pushforward`/`_value_and_gradient` via Enzyme (`AutoEnzyme`).
   - `GeoVIReactantExt.jl` runs the VI loop under Reactant for GPU/TPU; uses `AutoReactant`, wraps the rng into a `ReactantRNG` at `init`, and (in `_run_vi!`) `@compile`s `step_vi!` once then loops the compiled thunk. No persistent compile cache — `step_vi!` is itself the compilable primitive (users can `@compile` it directly).
   Both load automatically when their package is available.

### Reactant tracing convention (preserve when editing)

Hot loops use `ReactantCore.@trace` so they run under both eager and traced execution. This spans `src/cg.jl` (inner CG), `src/optimize.jl` (Newton/line-search loops), `src/vi.jl` (the Monte Carlo divergence sum), `src/families/fisher_gaussian.jl` (the `draw_samples!` de-whitening loop and the natural-gradient metric's MC sum), `src/families/interface.jl` (the antithetic-pair write loop in the default `draw_samples!`), and `ext/GeoVIReactantExt.jl`. Conventions to keep:

- Loops over heavyweight closures use `@trace track_numbers = false` so the body compiles to a single MLIR while/for body instead of `n` trace-time-unrolled copies. Unrolling each iteration's full forward model (e.g. a VLBI NUFT registers O(10) broadcast functions) compounds against Reactant's per-name uniquing cap of 10000.
- `track_numbers = false` keeps the type-walk away from plain `Int`/`Bool` fields in closure environments — so accumulators must already be traced (e.g. seed with `zero(eltype(position))`, not a literal `0`).
- A loop index `i` arrives as a `Reactant.TracedRNumber{<:Integer}` inside `@trace for`; index residual blocks through the `_sample_slice`/`_sample_position` helpers rather than plain getindex. See inline comments in `src/vi.jl` and `src/samples.jl`.
- A new loop-carried scalar must be a *distinct* `TracedRNumber`, not an alias of another carried value. E.g. `prev_value = value` (both already traced) makes two carry slots share one node and breaks `@trace while`'s arg/result matching; init it as `value + zero(value)` to force a fresh node. See `_optimize(::NewtonCG)` in `src/optimize.jl` (`prev_value`, the lagged energy for the CG `absdelta` coupling).
- Runtime branches on a *traced* value must use `@trace if` (or `ifelse`), never a plain `if`. A plain `if` is fine only when its condition is compile-time known — including type-level checks like `absdelta === nothing` (which stays plain on purpose: it gates whether the energy-criterion code is emitted at all, while the value comparison `energy_diff < absdelta` goes through `@trace if`). See `_cg_iterate` in `src/cg.jl`.
- Do NOT nest `@trace if` inside another `@trace if`, and do NOT reference a qualified module name (e.g. `ReactantCore.foo`) inside a `@trace if` body — the former fails macro expansion, the latter resolves in a local scope (`UndefVarError`). Hoist such checks above the `@trace if` into a plain `Bool`.

### The four axes (organizing principle)

A VI step is: **(1)** draw an MC sample set from `q` at the current mean, **(2)** move the mean to minimize the MC-estimated divergence. The knobs separate into four orthogonal axes passed directly to `VariationalProblem` (there is no `VIConfig`):

- **family** (`MGVIFamily` / `GeoVIFamily` / `MeanFieldGaussian`) — *how to draw one sample from `q`*, plus the solvers that draw needs. `MGVIFamily(; solver=ConjugateGradient(...))` does the linear metric draw; `GeoVIFamily(; solver=ConjugateGradient(...), curve=NewtonCG(...))` adds the nonlinear curve. `MeanFieldGaussian()` is mean-field ADVI (diagonal Gaussian `N(μ, diag σ²)`, reparam `ξ = μ + σ⊙ε`); it needs no solvers but **requires an `Optimisers.jl` rule** (e.g. `Optimisers.Adam`) and `n_samples > 0`, and ships its objective as the ELBO via the `(MeanFieldGaussian, ReverseKL)` joint dispatch. The family carries no outer-fit knobs.
- **estimator** (`MCEstimator(; n_samples, mirrored)`) — *how many MC nodes* estimate `E_q[·]`. Family-agnostic; the curve/CG tolerances are NOT here.
- **optimizer** (`NewtonCG(...)` or a bare `Optimisers.jl` rule) — the *outer position update only*. `NewtonCG` is a to-convergence solver carrying flat CG keywords (`cg_rtol`, …); a bare `Optimisers.jl` rule (e.g. `Optimisers.Adam(0.05)`) takes one gradient step per `step_vi!`, with the user's loop providing the iteration count. `NewtonCG` energy-based convergence is opt-in via `absdelta` (absolute) or `delta` (per-d.o.f.; `absdelta = delta·length(x0)` at solve time, cf. NIFTy.re) — setting either also enables the inner-CG energy-decrease coupling (`cg_absdelta = absdelta/100` first iter, `0.1·(last Newton gain)` after). Default `NewtonCG()` is uncoupled.
- **divergence** (`ReverseKL`; `ForwardKL` is a stub) — the objective form. `_fdivergence_value` dispatches jointly on `(family, divergence)` and is the single reverse-KL objective for *every* family (`mean_i[-log p(ξ_i) - logjac_i]`, where `(ξ_i, logjac_i) = transport_and_logjac(family, θ, r_i)`). The natural-gradient metric is **not** on this axis — it is `natural_gradient_metric(family, lh, θ, residuals, v)`, an optimizer concern used only by `NewtonCG`.

### Key types

- `Samples` — holds `position` (expansion point, or `nothing`) and `residuals`; `posterior_samples(s)` returns `position .+ residuals` (or just `residuals` when `position === nothing`, as the fitted posterior stores it — full reconstructed `ξ` samples)
- `VariationalProblem` — bundles likelihood + the four axes + adtype (immutable)
- `VIState` — the mutable *numeric* loop state allocated once by `init` (`position` = the variational parameters `θ` (a bare array, or a NamedTuple for mean-field), residual buffer, the family-defined `noise` bundle for sample reuse, optimizer state); advanced in place by `step_vi!`. It holds no host-only fields (no iteration counter, no compile cache), so `step_vi!` is a pure in-place mutation the user can `@compile` directly under Reactant. The problem and rng are kept *out* of it and passed to `step_vi!` separately
- `AbstractVariationalDistribution` — the fitted variational distribution `q_θ`, returned by `distribution(family, θ, lh)` / `distribution(problem, state)` / `fit`. A first-class, *pure* distribution (no retained samples): `rand(rng, q[, n])` is universal (a CG solve for the Fisher-Gaussian families), and `logdensity(q, ξ)` is optional (defined where the density is tractable). Built-ins live in the family files: `DiagonalGaussian` (mean-field — standalone `N(μ, diag σ²)`, has `logdensity`) and `FisherGaussianDistribution` (MGVI/geoVI shared — carries family + mean + likelihood, no `logdensity`). Each has `.mean`
- `ConjugateGradient` — the linear solver; user-facing as a family `solver`, internal inside `NewtonCG`
- `OptimizationResult` — returned by inner and outer optimizers with convergence info

### Extension points (from `docs/src/interfaces.md`)

- Custom likelihoods: subtype `AbstractLikelihood`, implement required methods
- Custom variational families (the families live in `src/families/`: `interface.jl` defines the contract + all generic defaults, then one file per family — `mgvi.jl`, `geovi.jl`, the shared `fisher_gaussian.jl`, `meanfield.jl`): subtype `AbstractVariationalFamily`, implement `init_params` + `transport_and_logjac` (the latter only if `θ` is not itself the latent point — its default is `(θ .+ r, 0)`). `transport_and_logjac(family, θ, r) -> (ξ, logjac)` returns the reconstructed sample and the reparameterization's log-Jacobian (the family's log q term, fused so a flow's single forward pass yields both). Each other hook has a default targeting the pushforward case: a family with a non-trivial frozen draw overrides `draw_samples!` (default: fill the residual buffer with IID white noise — antithetic ±ε pairs when `mirrored`); and to enable `NewtonCG`, set `supports_natural_gradient(family) = true` and implement `natural_gradient_metric`. MGVI/geoVI share one implementation via the `FisherGaussian` union and differ only in `_refine_residual` (geoVI adds the curve)
- Custom divergences: subtype `AbstractFDivergence`, implement `_fdivergence_value(family, divergence, …)` — joint `family × divergence` dispatch. (The metric is not on this axis; see `natural_gradient_metric`, an optimizer concern.)
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
q = distribution(problem, state)

# or the convenience driver:
q = fit(MersenneTwister(42), problem, 8)

# restart from a new point without rebuilding `problem`:
rng, state = init(MersenneTwister(42), problem, xi0_new)   # or: reset!(state, problem, xi0_new)

draws = rand(MersenneTwister(0), q, 100)      # draw arbitrarily many new samples
μ     = q.mean                                # the latent mean
```

Mean-field ADVI swaps the family for `MeanFieldGaussian()` and the optimizer for an
`Optimisers.jl` rule (no solvers; `θ = (; mean, logstd)` is a NamedTuple). The fitted output
is a `DiagonalGaussian` carrying `mean`/`logstd`; `rand(q)` reconstructs `μ + σ⊙ε`:

```julia
problem = VariationalProblem(lh, xi0;
    family    = MeanFieldGaussian(),
    estimator = MCEstimator(; n_samples=128, mirrored=true),
    optimizer = Optimisers.Adam(0.05),     # required: a bare Optimisers rule
)                                          # adtype defaults to AutoFiniteDiff (tree-generic via destructure)
q = fit(MersenneTwister(0), problem, 3000)
σ    = exp.(q.logstd)                      # recovered marginal std devs
```

Under Reactant (`xi0` a Reactant array → `AutoReactant`), `step_vi!` is the
compilable primitive — compile it once and loop the thunk (`fit` does this for
you):

```julia
rng, state = init(MersenneTwister(42), problem)   # rng wrapped into a ReactantRNG
cstep = Reactant.@compile step_vi!(rng, problem, state)
for _ in 1:8
    cstep(rng, problem, state)
end
q = distribution(problem, state)
```

Mean-field ADVI runs under Reactant too (the NamedTuple `θ` and the wrapped
multi-leaf optimizer state are handled). This needed one fix in the bare-`Optimisers`-rule
loop (`_run_optimizer_rule`, never compiled under Reactant before — existing Reactant
coverage is `NewtonCG` only): `_evaluate_optimizer_candidate` now evaluates `fun_and_grad`
**unconditionally** instead of inside `@trace if step_valid`. Guarding the Enzyme autodiff
behind a `@trace if` inside the loop's `@trace while` triggers a Reactant compiler bug —
XLA fails with `operand #N does not dominate this use` (the trace succeeds; compilation
does not). Bisected: reverting only this helper to the conditional form re-breaks it, and
a *trivial* (non-autodiff) `fun_and_grad` compiles, so it is the conditional autodiff
specifically. The other `@trace if`s in the loop (`_advance_optimizer_state` iterate
rebind, the check helpers) are fine and unchanged.

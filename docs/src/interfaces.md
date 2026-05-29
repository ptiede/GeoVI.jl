# Interfaces

This page records the currently supported extension points in `GeoVI.jl`.
The package is still intentionally small, so these interfaces are explicit
rather than abstracted behind a large trait system.

## Likelihoods

All likelihoods subtype `AbstractLikelihood` and live in the observation space
where the Fisher geometry is naturally defined.

Required methods for a new likelihood `MyLikelihood <: AbstractLikelihood`:

```julia
logdensity(lh::MyLikelihood, y)
normalized_residual(lh::MyLikelihood, y)
transformation(lh::MyLikelihood, y)
leftsqrtmetric(lh::MyLikelihood, y, η)
rightsqrtmetric(lh::MyLikelihood, y, v)
```

Optional methods:

```julia
fishermetric(lh::MyLikelihood, y, v)
```

If `fishermetric` is omitted, GeoVI uses the default composition
`leftsqrtmetric(lh, y, rightsqrtmetric(lh, y, v))`.

Notes:

- `logdensity` is interpreted as an unnormalized log density or log likelihood,
  up to additive constants independent of `y`.
- `energy(lh, y)` is retained as a compatibility alias for `-logdensity(lh, y)`.
- `metric(lh, y, v)` is retained as a compatibility alias for
  `fishermetric(lh, y, v)`.

## Variational Families

A variational family defines *how a sample is drawn from the variational
distribution* `q`, and carries the solvers that draw needs (but no
outer-optimization or Monte-Carlo-budget knobs):

```julia
abstract type AbstractVariationalFamily end
```

The built-in families are `MGVIFamily(; solver=ConjugateGradient(...))` and
`GeoVIFamily(; solver=ConjugateGradient(...), curve=NewtonCG(...))`.

To add a new family, subtype `AbstractVariationalFamily` and implement

```julia
_draw_sample_block(family::YourFamily, lh, position, rng, mirrored)
_draw_one_residual(family::YourFamily, lh, position, rng)
```

The first returns a (possibly mirrored) sample block plus auxiliary sampler
info and is used during fitting; the second returns a single residual and backs
`rand` on a `VariationalPosterior`. Both consume the family's own stored solvers.

## Estimators

The Monte-Carlo estimator owns how many sample nodes approximate the divergence
expectation `E_q[·]` — a property of the expectation, not the family:

```julia
abstract type AbstractEstimator end
```

The built-in estimator is `MCEstimator(; n_samples, mirrored)`. To add a new
estimator, subtype `AbstractEstimator` and provide `_n_base_draws` and a
`draw_residuals(family, estimator, lh, position, rng)` method.

## Divergences

The current divergence surface is:

```julia
abstract type AbstractFDivergence end
```

To add a new divergence, subtype `AbstractFDivergence` and implement, dispatching
**jointly on the family and the divergence**:

```julia
_fdivergence_value(family, ::YourDivergence, lh, position, residuals)
_fdivergence_fishermetric(family, ::YourDivergence, lh, position, residuals, v)
```

The first method provides the scalar objective minimized by `fit`, and the
second provides the associated Fisher-metric action used by second-order
optimizers. The joint `family × divergence` dispatch lets a scheme ship its own
objective.

## Optimizers

GeoVI accepts, as the outer position optimizer:

- built-in optimizers that subtype `AbstractOptimizer` (`NewtonCG`), which run to
  convergence within one position update. `NewtonCG` enables energy-based
  convergence (and the inner-CG energy-decrease coupling) when given `absdelta`
  (absolute) or `delta` (per-degree-of-freedom; `absdelta = delta·length(x0)` at
  solve time, since the energy is a sum over latent dimensions)
- a bare `Optimisers.AbstractRule` (e.g. `Optimisers.Adam(0.05)`), which takes a
  single gradient step per `step_vi!` — the user's loop provides the iterations

To add a new built-in optimizer, subtype `AbstractOptimizer` and implement:

```julia
_optimize(
    optimizer::YourOptimizer,
    x0::AbstractArray;
    fun_and_grad,
    metricp,
    maxiter,
    miniter,
    xtol,
    absdelta,
    cg_rtol,
    cg_atol,
    cg_maxiter,
    cg_miniter,
    stepnorm,
    optimizer_state,
)
```

The return value should be an `OptimizationResult`.

If the backend carries state across outer VI iterations, also implement:

```julia
_optimizer_state(optimizer::YourOptimizer, x0, previous_result)
```

For simple stateless optimizers, the default `_optimizer_state` method is
enough.

## Problem Setup

A `VariationalProblem` bundles the likelihood with the four orthogonal axes
(`family`, `divergence`, `estimator`, `optimizer`) plus the AD backend:

```julia
problem = VariationalProblem(lh, xi0; family, divergence, estimator, optimizer, adtype)
```

The primary loop is in place: the user owns the iteration count.

```julia
rng, state = init(rng, problem)   # state: one mutable, fully preallocated VIState
for _ in 1:n
    step_vi!(rng, problem, state) # mutates state and its buffers in place
end
post = posterior(problem, state)
```

`fit(problem, n; rng)` is a convenience that runs the loop and returns the
`VariationalPosterior`. `VariationalProblem` resolves the AD backend once (e.g.
inferring `AutoReactant` from a Reactant array position). Under Reactant the
in-place step is compiled, tracing the mutation directly.

A VI step is the reparameterization structure of VI sliced into three phases,
exposed as composable (unexported) primitives:

- `GeoVI.sample!(rng, problem, state)` — draw white noise `ξ_w ∼ N(0, I)`. The
  only stochastic phase.
- `GeoVI.transform!(problem, state)` — de-whiten: apply the family transform at
  the current mean (MGVI: CG solve; geoVI: CG + nonlinear curve) to turn the
  stored noise into sample residuals. Re-running it (without `sample!`) recomputes
  the Fisher/transform at the moved mean for the same realization.
- `GeoVI.update!(problem, state)` — estimate the KL and move the variational mean.

`step_vi!` runs `sample!` → `transform!` → `update!`. For custom schedules — e.g.
draw once, then refine the mean against a fixed realization (recompute-Fisher) —
compose the phases directly:

```julia
rng, state = init(rng, problem)
GeoVI.sample!(rng, problem, state)
for _ in 1:k
    GeoVI.transform!(problem, state)   # recompute the Fisher at the moved mean
    GeoVI.update!(problem, state)
end
```

## AD Backends

Automatic differentiation backends are selected with `ADTypes.jl`.

To add a new backend, implement:

```julia
_automatic_linearize(adtype::YourADType, forward, x; fd_eps=1e-6)
_value_and_gradient(adtype::YourADType, objective, x; fd_eps=1e-6)
```

`_automatic_linearize` should return a named tuple with fields:

```julia
(value=value_at_x, pushforward=jvp, pullback=vjp)
```

where:

- `pushforward(v)` computes `J(x) * v`
- `pullback(η)` computes `J(x)' * η`

Backends that need array-specific selection can additionally implement:

```julia
_infer_adtype(adtype, x)
_infer_composed_adtype(adtype, x)
```

That pattern is used by the Enzyme and Reactant extensions to pick a backend
that matches the storage type of `x`.

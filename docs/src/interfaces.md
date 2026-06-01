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

A variational family is the distribution `q_θ`. VI minimizes the reverse KL

```
J(θ) = mean_i[ -log p(ξ_i) + log q_θ(ξ_i) ],   ξ_i = T_θ(η_i),  η_i ∼ N(0, I).
```

```julia
abstract type AbstractVariationalFamily end
```

The built-in families are `MGVIFamily(; solver=ConjugateGradient(...))`,
`GeoVIFamily(; solver=ConjugateGradient(...), curve=NewtonCG(...))`, and
`MeanFieldGaussian()` (mean-field ADVI — a diagonal Gaussian `N(μ, diag σ²)` with the
reparameterization `ξ = μ + σ⊙ε`).

### The frozen prefix / differentiated suffix split

Families differ on **one axis**: where the reparameterization `T_θ` is cut into a part
*frozen* at the current `θ` and a part *differentiated* through `θ`. That cut is the two
hooks `transform_block` (frozen) and `transport` (differentiated):

- `transform_block` — the frozen prefix, run at the current `θ` and held constant,
  producing a stored residual. Trivial (`= ε`) for pushforward families; a CG solve
  (`MGVIFamily`) or CG + nonlinear curve (`GeoVIFamily`) for the Fisher-Gaussian families.
  Expensive, *not* differentiated, reusable for common random numbers.
- `transport` — the differentiated suffix, reconstructing `ξ` from `θ` + the stored
  residual. `MGVI`/`geoVI`: `θ + r`; mean-field: `μ + σ⊙ε`; a flow: `T_θ(ε)`.

Everything else *follows* from the cut: freezing the covariance ⇒ the `log q` term is `0`
(the Fisher-Gaussian fixed-metric approximation); differentiating the shape ⇒ `log q`
carries `-Σ logσ` / `-log|det J|`. The natural-gradient metric is the frozen covariance
`(I+Fisher)` reused as the `NewtonCG` preconditioner — which is why `NewtonCG` is
available for the Fisher-Gaussian families and not the pushforward ones.

### Adding a family

Subtype `AbstractVariationalFamily` and implement the **required** methods:

```julia
GeoVI.init_params(family, latent)  # the parameter container θ from the user's initial ξ₀
GeoVI.transport(family, θ, r)        # reconstruct ξ from θ + a residual (DIFFERENTIATED through θ)
```

`transport` defaults to `θ .+ r` (so a bare-array `θ`, like MGVI/geoVI, needs no override).
Each **optional** hook has a default targeting the pushforward case (mean-field,
normalizing flows), so a new pushforward family typically adds only:

```julia
GeoVI.logdensity(family, θ, ε)  # the family's log-density log q_θ(ξ); default 0 (fixed metric)
```

A *structured* `θ` (e.g. `θ = (; mean, logstd)`) — being any Functors-compatible
container — works with `Optimisers.setup`/`update` and the Enzyme/Reactant AD backends
natively; such a family overrides `transport` (and, to back `rand`,
`GeoVI.draw_noise(family, lh, θ, rng)`). A family with a non-trivial *frozen* draw
overrides `GeoVI.transform_block(family, lh, θ, noise_i, mirrored)` and
`GeoVI.init_noise(family, adtype, lh, latent, n)` (whose default is a single latent-shaped
`ε` buffer; the noise bundle may be any Functors-traversable structure of arrays). To be
usable with the `NewtonCG` optimizer, also set `GeoVI.supports_natural_gradient(family) =
true` and implement `GeoVI.natural_gradient_metric(family, lh, θ, residuals, v)`.

A complete minimal pushforward family is just the three required/optional methods above —
no solver, no metric, no buffer plumbing.

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

To add a new divergence, subtype `AbstractFDivergence` and implement the scalar objective
minimized by `fit`, dispatching **jointly on the family and the divergence**:

```julia
_fdivergence_value(family, ::YourDivergence, lh, position, residuals)
```

The joint `family × divergence` dispatch lets a scheme ship its own objective form. The
natural-gradient metric used by second-order optimizers is *not* part of the divergence
surface — it is an optimizer concern carried by the family
(`GeoVI.natural_gradient_metric`, consulted only by `NewtonCG`); see *Variational
Families* above.

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
rng, state = init(rng, problem)        # state: one mutable, fully preallocated VIState
for _ in 1:n
    step_vi!(rng, problem, state)      # mutates state and its buffers in place
end
q = distribution(problem, state)
```

`step_vi!(rng, problem, state, n_refine=0)` takes an optional fourth argument:
after the fresh `sample!` → `transform!` → `update!` cycle it runs `n_refine`
extra `transform!` → `update!` refinements that **reuse the drawn noise**
(recompute the Fisher / re-curve at the moved mean — common random numbers).

`fit(problem, n; rng, n_refine=0)` is a convenience that runs the loop and
returns the fitted variational distribution. `VariationalProblem` resolves the AD backend
once (e.g. inferring `AutoReactant` from a Reactant array position).

Under Reactant, `step_vi!` is a pure in-place mutation with no host-only state,
so you compile it yourself and loop the compiled thunk. Passing `n_refine` as a
`ConcreteRNumber{Int}` keeps it a runtime loop bound, so one compiled graph
serves any refinement count:

```julia
rng, state = init(rng, problem)        # rng is wrapped into a ReactantRNG here
nref  = Reactant.ConcreteRNumber(k)
cstep = Reactant.@compile step_vi!(rng, problem, state, nref)
for _ in 1:n
    cstep(rng, problem, state, nref)
end
```

`fit` does exactly this for you on the Reactant path (compile once, loop the
thunk).

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

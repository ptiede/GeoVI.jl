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
hooks `draw_samples!` (frozen) and `transport_and_logjac` (differentiated):

- `draw_samples!` — the frozen prefix: fill the residual buffer with the Monte-Carlo set
  drawn at the current `θ` and held constant during the update. IID white noise for
  pushforward families; a CG solve (`MGVIFamily`) or CG + nonlinear curve (`GeoVIFamily`)
  for the Fisher-Gaussian families. Expensive, *not* differentiated, redrawn each step.
- `transport_and_logjac` — the differentiated suffix, returning `(ξ, logjac)`: the
  reconstructed `ξ` *and* the reparameterization's log-Jacobian `log|det J|`. `MGVI`/`geoVI`:
  `(θ + r, 0)`; mean-field: `(μ + σ⊙ε, Σ logσ)`; a flow: `(T_θ(ε), log|det J_θ|)` — one
  forward pass yields both.

Everything else *follows* from the cut: the reverse-KL objective is
`mean_i[-log p(ξ_i) - logjac_i]`. Freezing the covariance ⇒ `logjac = 0` (the
Fisher-Gaussian fixed-metric approximation — the intractable metric log-det is dropped);
differentiating the shape ⇒ `logjac` carries `Σ logσ` / `log|det J|`. The natural-gradient
metric is the frozen covariance `(I+Fisher)` reused as the `NewtonCG` preconditioner —
which is why `NewtonCG` is available for the Fisher-Gaussian families and not the
pushforward ones.

### Adding a family

Subtype `AbstractVariationalFamily` and implement the **required** methods:

```julia
GeoVI.init_params(family, latent)            # the parameter container θ from the user's initial ξ₀
GeoVI.transport_and_logjac(family, θ, r)     # -> (ξ, logjac): reconstruct ξ AND its log-Jacobian
```

`transport_and_logjac` defaults to `(θ .+ r, 0)` (so a bare-array `θ`, like MGVI/geoVI,
needs no override). It returns both the sample and the reparameterization's log-Jacobian
(the family's `log q` term) so that, e.g., a normalizing flow produces `ξ` and its
`log|det J|` from a single forward pass.

A *structured* `θ` (e.g. `θ = (; mean, logstd)`) — being any Functors-compatible
container — works with `Optimisers.setup`/`update` and the Enzyme/Reactant AD backends
natively; such a family overrides `transport_and_logjac` (and its own distribution type's
`rand`). A family with a non-trivial *frozen* draw overrides
`GeoVI.draw_samples!(family, lh, θ, residuals, rng, mirrored)` (whose default fills the
buffer with IID white noise, writing antithetic ±ε pairs when `mirrored`). To be usable
with the `NewtonCG` optimizer, also set `GeoVI.supports_natural_gradient(family) = true`
and implement `GeoVI.natural_gradient_metric(family, lh, θ, residuals, v)`. The metric
reaches the optimizer as a `NaturalGradientField` — the metric *field* over θ-space,
which `NewtonCG` evaluates at the current base point once per Newton iteration. Pinning
the field at a base point goes through `GeoVI._natural_gradient_operator(family, lh,
base, residuals)`, whose default returns the re-deriving `NaturalGradientOperator`; a
family may override it to hoist per-base work out of the inner-CG matvec loop (the
Fisher-Gaussian families cache every sample point's forward-model linearization there).

A complete minimal pushforward family is just `init_params` + `transport_and_logjac` —
no solver, no metric, no buffer plumbing.

## Estimators

The Monte-Carlo estimator owns how many sample nodes approximate the divergence
expectation `E_q[·]` — a property of the expectation, not the family:

```julia
abstract type AbstractEstimator end
```

The built-in estimator is `MCEstimator(; n_samples=4, mirrored=true)`. The VI
loop consults the estimator only through three accessors — never struct fields —
so a custom estimator subtypes `AbstractEstimator` and implements:

```julia
GeoVI._n_stored_samples(est)  # rows in the residual buffer (mirrored pairs count as two)
GeoVI._mirrored(est)          # whether the rows are antithetic ±pairs
GeoVI._n_base_draws(est)      # independent base draws
```

`n_samples = 0` (sample-free MAP) is only valid for families whose parameters
are themselves the latent point (MGVI/geoVI); structured-θ families require
samples. The standalone `draw_residuals(family, estimator, lh, position, rng)`
helper uses the same accessors.

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
  convergence within one position update. `NewtonCG` is *inexact* Newton: its
  inner CG stops at the Eisenstat–Walker forcing threshold `min(0.5, √‖g‖)·‖g‖`
  derived from the current gradient. `cg_rtol`/`cg_atol` default to `nothing`
  (forcing alone); when set they can only **tighten** the inner solve, via
  `min(forcing, max(cg_atol, cg_rtol·‖g‖))` — to spend *less* inner effort use
  `cg_maxiter` or the energy coupling instead. Energy-based convergence (and the
  inner-CG energy-decrease coupling) is enabled by `absdelta` (absolute) or
  `delta` (per-degree-of-freedom; `absdelta = delta·length(x0)` at solve time,
  since the energy is a sum over latent dimensions)
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

The return value should be an `OptimizationResult`. `metricp` is a metric *field*
over the parameter space (e.g. a `NaturalGradientField`): `metricp(x)` pins the
field at the base point `x`, returning the operator `v -> M(x)·v` on the tangent
space there — evaluate it once per outer iteration and reuse the pinned operator
for every application (inner-CG matvecs, line-search curvature), since pinning
may cache expensive per-base work such as a forward-model linearization (see
`GeoVI._at_point`).

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

`fit([rng], problem, n)` is a convenience that runs the loop and returns the fitted
variational distribution (rng positional or auto). `reset!(state, problem, ξ0)`
re-initializes an existing state in place from a new starting point (reuses buffers;
Reactant-safe). `VariationalProblem` resolves the AD backend once (e.g. inferring
`AutoReactant` from a Reactant array position).

Under Reactant, `step_vi!` is a pure in-place mutation with no host-only state,
so you compile it yourself and loop the compiled thunk:

```julia
rng, state = init(rng, problem)        # rng is wrapped into a ReactantRNG here
cstep = Reactant.@compile step_vi!(rng, problem, state)
for _ in 1:n
    cstep(rng, problem, state)
end
```

`fit` does exactly this for you on the Reactant path (compile once, loop the
thunk).

A VI step is the `draw → optimize-against-fixed-samples → resample` loop (the same
one NIFTy's `OptimizeVI.update` runs), sliced into two phases exposed as composable
(unexported) primitives:

- `GeoVI.draw_samples!(rng, problem, state)` — fill `state.residuals` with a fresh
  Monte-Carlo set drawn at the current mean (pushforward: IID white noise; MGVI: CG
  solve; geoVI: CG + nonlinear curve). The only stochastic phase.
- `GeoVI.update!(problem, state)` — estimate the KL with that fixed set and move the
  variational parameters (one `Optimisers` step, or `NewtonCG` to convergence).

`step_vi!` runs `draw_samples!` → `update!`. For a custom schedule, compose the two
phases directly:

```julia
rng, state = init(rng, problem)
for _ in 1:n
    GeoVI.draw_samples!(rng, problem, state)
    GeoVI.update!(problem, state)
end
```

## AD Backends

Automatic differentiation backends are selected with `ADTypes.jl`.

!!! warning "The finite-difference default is O(D) more expensive"
    The default `adtype = AutoFiniteDiff()` needs no extra packages, but each
    gradient costs `2·length(ξ₀)` objective evaluations — and each objective
    evaluation is an `n_samples` Monte-Carlo sum over the forward model. For
    anything beyond toy problems, load Enzyme and pass `adtype = AutoEnzyme()`
    (automatically upgraded to `AutoReactant` for Reactant arrays).

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

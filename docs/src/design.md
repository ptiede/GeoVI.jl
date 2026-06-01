# Design Proposal

This page captures the first-pass package shape for `GeoVI.jl`.
It is intentionally interface-first: the goal is to agree on names,
abstractions, and implementation order before we commit to the full numerical
machinery.

The proposal is based on:

- Frank, Leike, Ensslin, "Geometric variational inference" (Algorithm 2 and
  the sampling construction in Sections 3.1-3.2)
- `nifty.re`, especially `likelihood.py`, `evi.py`, and `optimize_kl.py`

## Design Goals

- Stay close to the paper's white-latent formulation, where the prior is
  standard normal and the posterior geometry is carried by the likelihood.
- Preserve the useful separation already present in `nifty.re`:
  likelihood, forward model composition, residual sampling, and outer VI loop.
- Be Julia-first in surface API:
  callables instead of Python classes where possible, immutable config/state
  objects, and keyword-driven construction.
- Keep Reactant and Enzyme as execution details of the numerics, not as noise
  in the user-facing API.
- Make MGVI and geoVI live in the same framework, with geoVI reducing to MGVI
  when the nonlinear residual update is disabled.

## Scope Assumption For The MVP

The first implementation pass should assume the latent variable `xi` lives in
standard normal coordinates:

```math
\xi \sim \mathcal{N}(0, I).
```

This matches the paper and the structure used in `nifty.re`. In practice, the
user expresses hierarchical priors by providing a forward or generative map
from `xi` into model or data space rather than by supplying an arbitrary prior
object.

That gives us a clean first target:

- posterior objective: `-logdensity(likelihood, forward(xi)) + 0.5 * dot(xi, xi)`
- posterior Fisher metric: `I + Fisher pullback through forward`
- MGVI: linear residual samples from the inverse Fisher metric
- geoVI: nonlinear residual update via the metric-induced transformation

## Proposed Package Layers

### 1. Dense-array utilities

Internal dense-array arithmetic utilities:

- `randn_like`
- `norm`

For the current scope, the latent state is a single dense array in white
coordinates. We do not attempt to support tuples, named tuples, or other
structured latent containers in the MVP.

### 2. Likelihood interface

The core object should represent a likelihood in the space where the Fisher
Fisher geometry is naturally defined. A forward model can then be composed onto it.

Proposed minimal interface:

```julia
abstract type AbstractLikelihood end

logdensity(lh::AbstractLikelihood, y)
normalized_residual(lh::AbstractLikelihood, y)
transformation(lh::AbstractLikelihood, y)

leftsqrtmetric(lh::AbstractLikelihood, y, eta)
rightsqrtmetric(lh::AbstractLikelihood, y, v)
fishermetric(lh::AbstractLikelihood, y, v)

compose(lh::AbstractLikelihood, forward)
```

Intended semantics:

- `logdensity(lh, y)` is the unnormalized log likelihood at a prediction `y`
- `transformation(lh, y)` is the local coordinate map whose Jacobian induces
  the Fisher metric
- `leftsqrtmetric` and `rightsqrtmetric` are the square-root actions used by
  sampling
- `fishermetric` defaults to `leftsqrtmetric(rightsqrtmetric(...))`
- `compose(lh, forward)` builds a latent-space likelihood by pullback through
  the forward model
- `compose` is best thought of as using a single local linearization of the
  forward model that provides both `pushforward` and `pullback`
- if manual linearization data are not provided, `compose` can synthesize those
  actions automatically from an `adtype`

The first concrete likelihoods should be:

```julia
GaussianLikelihood(data; precision, sqrt_precision=nothing)
PoissonLikelihood(data; weight=1)
BernoulliLikelihood(data; weight=1)
BinomialLikelihood(data; trials, weight=1)
```

This is the easiest way to validate the whole stack against Wiener filtering
style examples, then move into common canonical exponential-family cases before
adding heavier-tailed or variable-covariance models.

### 3. Forward-model composition

NIFTy's `Likelihood.amend(...)` is useful, but in Julia I would prefer a more
explicit name:

```julia
lh_latent = compose(lh_data, forward)
```

or, if we want a constructor:

```julia
lh_latent = ComposedLikelihood(lh_data, forward)
```

The `forward` object itself should just be any callable. If later we want
shape metadata or an initializer, we can add a light wrapper, but I would not
make `Model` a required abstraction in v0.

For manual control, `compose` can accept either a bundled `linearize(x)` method
returning `value`, `pushforward`, and `pullback`, or explicit
`pushforward`/`pullback` callables. The preferred default is that `compose`
works with just `forward` plus an `adtype`, and only falls back to manual
linearization data when needed for performance or unsupported code paths.

## Public Data Containers

### Samples

We should keep the same useful convention as `nifty.re`: store residuals
relative to an expansion point and materialize posterior samples only when
requested.

```julia
struct Samples{P,S,K}
    position::P
    residuals::S
    keys::K
end

posterior_samples(samples::Samples)
Base.length(samples::Samples)
Base.getindex(samples::Samples, i::Int)
Base.iterate(samples::Samples, state...)
recenter(samples::Samples, new_position)
```

Notes:

- `position` is the expansion point / latent mean, or `nothing` (the fitted posterior stores `position = nothing` and `residuals` = the full reconstructed `ξ` samples)
- `residuals` are stored relative to `position` (or are the samples themselves when `position === nothing`)
- `keys` holds RNG seeds or sampler state, when relevant
- `posterior_samples(samples)` returns `position .+ residuals`, or just `residuals` when `position === nothing`
- `position` and `residuals` are dense arrays (or `position` is `nothing`)

### Four orthogonal axes

The VI step decomposes into four orthogonal axes, passed directly to
`VariationalProblem` (there is no monolithic config object). Each axis is a small
self-describing object:

```julia
# (1) family: how to draw one sample from q, plus the solvers the draw needs.
abstract type AbstractVariationalFamily end
struct MGVIFamily{S}   <: AbstractVariationalFamily; solver::S; end          # linear draw
struct GeoVIFamily{S,C} <: AbstractVariationalFamily; solver::S; curve::C; end # + nonlinear curve

# (2) estimator: how many Monte-Carlo nodes estimate E_q[·] (family-agnostic).
abstract type AbstractEstimator end
struct MCEstimator <: AbstractEstimator; n_samples::Int; mirrored::Bool; end

# (3) optimizer: the OUTER position update only.
abstract type AbstractOptimizer end
struct NewtonCG{T,A,C} <: AbstractOptimizer end       # carries flat CG keywords
# a bare Optimisers.jl rule is also accepted: one gradient step per step_vi!

# (4) divergence: the objective form; dispatches jointly with the family.
abstract type AbstractFDivergence end
struct ReverseKL <: AbstractFDivergence end
struct ForwardKL <: AbstractFDivergence end

struct VariationalProblem{L,S,F,D,E,O,AD}
    likelihood::L
    initial_samples::S
    family::F
    divergence::D
    estimator::E
    optimizer::O
    adtype::AD
end
```

The key separation: the linear-draw CG tolerance and the geoVI curve are
*family-specific draw machinery* (they live on the family); `n_samples`/`mirrored`
are *Monte-Carlo estimator* knobs (they live on `MCEstimator`); and the outer
`optimizer` governs only the position update. `ConjugateGradient` is user-facing
only as a family `solver`; inside `NewtonCG` the CG is configured by flat keywords.

Important departure from `nifty.re`:

- `n_samples` means the final number of stored samples, not the number of random
  seeds before mirroring.
- if `mirrored=true`, `MCEstimator` requires `iseven(n_samples)` and internally
  uses `n_samples ÷ 2` seeds.

## Public Algorithm Entry Points

### Low-level sampling kernels

These mirror the paper and `nifty.re`, and give us a clean way to test the
pieces independently.

```julia
draw_metric_sample(lh, xi, rng)
draw_linear_residual(lh, xi, rng_or_metric_sample; kwargs...)
update_nonlinear_residual(lh, xi, linear_draw; kwargs...)
draw_residual(lh, xi, rng; kwargs...)
```

Intended meaning:

- `draw_linear_residual` is the MGVI residual draw
- `update_nonlinear_residual` curves an existing residual into a geoVI sample
- `draw_residual` is a convenience wrapper that performs both

### Outer VI loop

The primary interface is the in-place loop (the user owns the iteration count):

```julia
problem = VariationalProblem(lh, xi0; family, divergence, estimator, optimizer, adtype)
rng, state = init(rng, problem)   # state: one mutable, fully preallocated VIState
for _ in 1:n
    step_vi!(rng, problem, state) # mutates state and its buffers in place
end
q = distribution(problem, state)  # an AbstractVariationalDistribution
```

`step_vi!(rng, problem, state)` runs one `draw_samples!`→`update!` cycle: draw a
Monte-Carlo set at the current mean, then optimize the variational parameters against
that fixed set (the `draw → optimize-against-fixed-samples → resample` loop).

`fit(problem, n; rng)` is a convenience that runs the loop and returns the posterior.
Under Reactant, `step_vi!` is a pure in-place mutation the user `@compile`s themselves
and loops; `fit` does this for you (compile `step_vi!` once, then loop the thunk).

## Example User Flow

This is the surface API we target:

```julia
using GeoVI
using ADTypes
using Optimisers

forward(xi) = A * exp.(xi)

lh = GaussianLikelihood(data; precision = inv_noise_cov)
posterior_lh = compose(lh, forward)

problem = VariationalProblem(
    posterior_lh,
    xi0;
    family    = GeoVIFamily(; solver = ConjugateGradient(rtol = 1e-4, maxiter = 100),
                             curve  = NewtonCG(maxiter = 5, xtol = 1e-4)),
    divergence= ReverseKL(),
    estimator = MCEstimator(; n_samples = 8, mirrored = true),
    optimizer = NewtonCG(maxiter = 35, xtol = 1e-4),
    adtype    = ADTypes.AutoEnzyme(),
)

q     = fit(problem, 8; rng)        # returns the fitted variational distribution
draws = rand(rng, q, 100)           # draw arbitrarily many new samples
μ     = q.mean                      # the latent mean
```

This remains valid whether the differentiation engine is finite differences,
Enzyme, or `ADTypes.AutoReactant()` (inferred from a Reactant array position).
An `Optimisers.jl` rule may be used as the outer optimizer directly
(`optimizer = Optimisers.Adam(0.05)`); it takes one gradient step per
`step_vi!`, so the iteration count is the user's outer loop.

## ADTypes + Reactant Positioning

The package should be designed so that the public API does not care which AD
engine is chosen. Instead of a custom backend trait, the user should specify an
`adtype` and we dispatch internally from that.

That keeps the surface cleaner:

- `family`, `divergence`, `estimator`, and `optimizer` are explicit algorithm
  choices passed directly to `VariationalProblem` (the four orthogonal axes;
  there is no `VIConfig` — the `estimator` holds sample counts, the family/optimizer
  hold their tolerances)
- `adtype` selects the differentiation engine via `ADTypes`

Reactant should still not require a separate public backend hierarchy. If the
initial point already lives in a Reactant array type, the internal AD dispatch
can infer `ADTypes.AutoReactant()` from that.

Internally we still need implementation hooks for:

- array conversion into Reactant buffers when desired
- compilation / caching of repeated kernels
- AD primitives used for pullbacks, pushforwards, and Fisher-metric actions
- batching / mapping across multiple sample draws

Proposed mental model:

- `adtype = ADTypes.AutoFiniteDiff()` is the reference fallback path
- `adtype = ADTypes.AutoEnzyme()` is the first real AD path
- `adtype = ADTypes.AutoReactant()` is the performance path once compilation is wired in

I would avoid exposing raw Enzyme activity wrappers (`Const`, `Duplicated`,
etc.) in the public API. Those should stay internal helper machinery.

## Implementation Order

### Phase 1: Foundations

- Dense-array utilities for white latent coordinates
- `AbstractLikelihood`, `GaussianLikelihood`, and `ComposedLikelihood`
- `Samples`, `VariationalProblem` (the four axes), `VIState`
- a reference finite-difference fallback

### Phase 2: MGVI core

- `draw_linear_residual`
- Fisher-metric action and inverse-metric CG solve
- mirrored sampling
- linear-Gaussian validation against analytic posterior moments

### Phase 3: geoVI core

- `transformation`-based nonlinear residual objective
- `update_nonlinear_residual`
- `draw_residual`
- toy nonlinear example matching the paper qualitatively

### Phase 4: Outer VI loop

- `step_vi!`
- `fit`
- sample reuse / resampling policy
- optimizer and divergence plumbing

### Phase 5: Reactant backend

- Reactant buffer conversion and kernel compilation
- Enzyme-based pullback / Fisher-metric actions that survive tracing
- performance checks against the reference backend

### Phase 6: Feature parity extensions

- point estimates and frozen constants
- additional likelihoods
- structured diagnostics and minisanity-like summaries
- checkpoint / resume

## Current Modeling Choice

The package now assumes:

- all latent parameters are represented as a single dense array
- those latent parameters live in IID standard normal coordinates
- hierarchical structure is handled by the forward / generative map, not by a
  structured latent container
- the top-level fitting API is `fit`, with configurable `family`, `optimizer`,
  and `divergence`

This keeps the implementation focused on the white-parameter geometry from the
paper and avoids carrying generic tree machinery that the intended use case
does not need.

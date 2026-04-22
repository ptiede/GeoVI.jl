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

- `position` is the expansion point / latent mean
- `residuals` are stored relative to `position`
- `keys` holds RNG seeds or sampler state, when relevant
- `posterior_samples(samples)` should return `position .+ residuals`
- both `position` and `residuals` are dense arrays in the current design

### VI configuration and state

Instead of a large hidden optimizer backend, I would make the algorithm mostly
functional and carry any optimizer state explicitly.

```julia
abstract type AbstractVariationalFamily end
struct MGVIFamily <: AbstractVariationalFamily end
struct GeoVIFamily <: AbstractVariationalFamily end

abstract type AbstractFDivergence end
struct ReverseKL <: AbstractFDivergence end
struct ForwardKL <: AbstractFDivergence end

abstract type AbstractOptimizer end
struct NewtonCG <: AbstractOptimizer end

struct VIConfig{AD,DL,NU,OO}
    adtype::AD
    n_iterations::Int
    n_samples::Int
    mirrored::Bool
    draw_linear::DL
    nonlinear_update::NU
    optimizer_options::OO
end

struct VIState{R,S,M}
    iteration::Int
    rng::R
    sample_state::S
    minimization_state::M
end

struct VariationalProblem{L,S,F,D,O,C,AD,DL,NU,OO}
    likelihood::L
    initial_samples::S
    family::F
    divergence::D
    optimizer::O
    config::C
    adtype::AD
    draw_linear_options::DL
    nonlinear_update_options::NU
    optimizer_options::OO
end
```

Here `optimizer` can be either our built-in `NewtonCG()` or a standard
`Optimisers.jl` rule such as `Optimisers.Adam()`.

Important proposed departure from `nifty.re`:

- `n_samples` should mean the final number of stored samples, not the number of
  random seeds before mirroring.
- if `mirrored=true`, require `iseven(n_samples)` and internally use
  `n_samples ÷ 2` seeds

That removes a very easy source of confusion from the Python implementation.

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

Proposed high-level driver:

```julia
problem = VariationalProblem(lh, xi0; family, divergence, optimizer, config)
initialize_vi(problem, rng)
step_vi(problem, samples, state)
fit(problem; rng)
```

This keeps `fit` as the public entry point without carrying older aliases.

## Example User Flow

This is the kind of surface API I think we should target first:

```julia
using GeoVI
using ADTypes
using Optimisers

forward(xi) = A * exp.(xi)

lh = GaussianLikelihood(data; precision = inv_noise_cov)
posterior_lh = compose(lh, forward)

cfg = VIConfig(
    adtype = ADTypes.AutoEnzyme(),
    n_iterations = 8,
    n_samples = 8,
    mirrored = true,
    draw_linear = (; cg_maxiter = 100, cg_tol = 1e-4),
    nonlinear_update = (; maxiter = 5, xtol = 1e-4),
    optimizer_options = (; maxiter = 35, xtol = 1e-4),
)

samples, state = fit(
    rng,
    posterior_lh,
    xi0,
    GeoVIFamily(),
    ReverseKL(),
    Optimisers.Adam(0.05);
    config = cfg,
)
draws = posterior_samples(samples)
```

This should remain valid whether the differentiation engine is finite
differences, Enzyme, or later `ADTypes.AutoReactant()`.

## ADTypes + Reactant Positioning

The package should be designed so that the public API does not care which AD
engine is chosen. Instead of a custom backend trait, the user should specify an
`adtype` and we dispatch internally from that.

That keeps the surface cleaner:

- `family`, `divergence`, and `optimizer` are explicit algorithm choices
- `VIConfig` holds tuning knobs like sample counts and tolerances
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
- `Samples`, `VIConfig`, `VIState`
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

- `step_vi`
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

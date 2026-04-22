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

Variational families are lightweight markers:

```julia
abstract type AbstractVariationalFamily end
```

To add a new family, subtype `AbstractVariationalFamily` and implement

```julia
_draw_sample_block(problem, ::YourFamily, position, rng)
```

returning a sample block and auxiliary sampler info.

## Divergences

The current divergence surface is:

```julia
abstract type AbstractFDivergence end
```

To add a new divergence, subtype `AbstractFDivergence` and implement:

```julia
_fdivergence_value(::YourDivergence, lh, position, residuals)
_fdivergence_fishermetric(::YourDivergence, lh, position, residuals, v)
```

The first method provides the scalar objective minimized by `fit`, and the
second provides the associated Fisher-metric action used by second-order
optimizers.

## Optimizers

GeoVI accepts two optimizer families:

- built-in optimizers that subtype `AbstractOptimizer`
- any `Optimisers.AbstractRule`

To add a new built-in optimizer, subtype `AbstractOptimizer` and implement:

```julia
_optimize(
    optimizer::YourOptimizer,
    x0::AbstractArray;
    fun_and_grad,
    hessp,
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

The preferred way to prepare the outer VI loop is:

```julia
problem = VariationalProblem(lh, xi0; family, divergence, optimizer, config)
state = initialize_vi(problem, rng)
samples, state = step_vi(problem, state)
samples, state = fit(problem; rng)
```

`VariationalProblem` resolves the AD backend and normalizes the relevant option
sets once, instead of repeating that work inside each VI iteration.

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

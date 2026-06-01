"""
    AbstractFDivergence

Marker supertype for f-divergence objectives optimized by the outer VI loop.

The objective hook `_fdivergence_value(family, divergence, ...)` dispatches jointly on
the family and the divergence, so a new scheme may ship its own objective form.
"""
abstract type AbstractFDivergence end
struct ReverseKL <: AbstractFDivergence end
struct ForwardKL <: AbstractFDivergence end

struct VariationalProblem{L, S, F, D, E, O, AD}
    likelihood::L
    initial_samples::S
    family::F
    divergence::D
    estimator::E
    optimizer::O
    adtype::AD
end

_problem_samples(samples::Samples) = samples
_problem_samples(position::AbstractArray) = Samples(position, nothing; keys = nothing)

_infer_adtype(adtype, x) = adtype

function _problem_adtype(adtype, samples::Samples)
    samples.position === nothing && return adtype
    return _infer_adtype(adtype, samples.position)
end

function VariationalProblem(
        lh::AbstractLikelihood,
        position_or_samples;
        family::AbstractVariationalFamily = GeoVIFamily(),
        divergence::AbstractFDivergence = ReverseKL(),
        estimator::AbstractEstimator = MCEstimator(),
        optimizer = NewtonCG(),
        adtype = ADTypes.AutoFiniteDiff(),
    )
    adtype === nothing && (adtype = ADTypes.AutoFiniteDiff())
    samples = _problem_samples(position_or_samples)
    _require_supported(family, divergence, estimator, optimizer)
    return VariationalProblem(
        lh,
        samples,
        family,
        divergence,
        estimator,
        optimizer,
        _problem_adtype(adtype, samples),
    )
end

_n_base_draws(problem::VariationalProblem) = _n_base_draws(problem.estimator)

# ── VI loop state ──────────────────────────────────────────────────────────

"""
    VIState

Mutable numeric state for the VI loop: only the evolving quantities, not the
problem or the RNG (both are passed to [`step_vi!`](@ref) separately). Allocated
once by [`init`](@ref) with every field at its final type — the current parameters
(`position`), the residual buffer (`residuals`) that [`draw_samples!`](@ref GeoVI.draw_samples!)
fills, and the threaded `optimizer_state` — advanced in place, so reusing one `VIState`
keeps per-iteration allocation flat. It holds no host-only fields (no iteration counter,
no compile cache), so [`step_vi!`](@ref) is a pure in-place mutation the user can
`@compile` directly under Reactant.

`residuals` is shaped `(n_samples, latent…)`, or `nothing` when there are no samples (MAP).
The white noise the draw consumes is transient scratch allocated inside `draw_samples!`,
not stored here.
"""
mutable struct VIState{P, R, Os}
    position::P
    residuals::R
    optimizer_state::Os
end

_wrap_rng(_adtype, rng) = rng

function _init_residual_buffer(problem::VariationalProblem, latent)
    problem.estimator.n_samples == 0 && return nothing
    return similar(latent, (problem.estimator.n_samples, size(latent)...))
end

# Fresh optimizer state for the chosen optimizer (`nothing` for the stateless
# `NewtonCG`, an `Optimisers` setup for a rule). Threaded across steps.
_init_optimizer_state(problem::VariationalProblem, position) =
    _optimizer_state(problem.optimizer, position, nothing)

"""
    init([rng], problem) -> (rng, state)

Allocate the [`VIState`](@ref) for `problem` — a fresh copy of the initial
position, the residual buffer, and the optimizer state —
and return it together with the loop RNG to thread through [`step_vi!`](@ref). For
an `AutoReactant` problem the RNG is wrapped into a `Reactant.ReactantRNG` (the
compiled step is built lazily by `fit`, or by the user calling
`@compile step_vi!(...)`). `rng` defaults to `Random.default_rng()`.
"""
function init(rng::AbstractRNG, problem::VariationalProblem)
    θ = init_params(problem.family, problem.initial_samples.position)
    # All buffers are sized from the user's initial latent point ξ₀ (latent-shaped by
    # construction), so the interface needs no `θ → latent` projection.
    latent = problem.initial_samples.position
    residuals = _init_residual_buffer(problem, latent)
    optimizer_state = _init_optimizer_state(problem, θ)
    wrapped_rng = _wrap_rng(problem.adtype, rng)
    return wrapped_rng, VIState(θ, residuals, optimizer_state)
end

init(problem::VariationalProblem; rng::AbstractRNG = Random.default_rng()) =
    init(rng, problem)

# ── Sample-block plumbing (Reactant-safe) ──────────────────────────────────

function _single_sample_block(residual::AbstractArray)
    return reshape(residual, (1, size(residual)...))
end

function _write_sample_block!(dest::AbstractArray, i, block::AbstractArray)
    block_size = size(block, 1)
    offset = (i - 1) * block_size
    trailing = ntuple(_ -> Colon(), max(ndims(block) - 1, 0))
    for j in 1:block_size
        dest[offset + j, trailing...] = block[j, trailing...]
    end
    return dest
end

_draw_linear_kwargs(solver::ConjugateGradient) = (;
    cg_rtol = solver.rtol,
    cg_atol = solver.atol,
    cg_maxiter = solver.maxiter,
    cg_miniter = solver.miniter,
)


"""
    draw_residuals(problem, position, rng)
    draw_residuals(family, estimator, lh, position, rng)

Draw the Monte-Carlo residual set used to estimate `E_q[·]` at the latent point
`position`: `estimator` controls the count/mirroring, `family` controls how each draw is
realized (via `GeoVI.draw_samples!`).
"""
draw_residuals(problem::VariationalProblem, position::AbstractArray, rng::AbstractRNG) =
    draw_residuals(problem.family, problem.estimator, problem.likelihood, position, rng)

function draw_residuals(
        family::AbstractVariationalFamily,
        estimator::MCEstimator,
        lh::AbstractLikelihood,
        position::AbstractArray,
        rng::AbstractRNG,
    )
    n = _n_base_draws(estimator)
    if n == 0
        return Samples(position, nothing; keys = nothing),
            (family = family, mirrored = estimator.mirrored, n_draws = 0)
    end

    residuals = similar(position, (estimator.n_samples, size(position)...))
    draw_samples!(family, lh, position, residuals, rng, estimator.mirrored)

    return Samples(position, residuals; keys = Base.OneTo(n)),
        (family = family, mirrored = estimator.mirrored, n_draws = n)
end

# ── The two VI phases (composable primitives; unexported) ───────────────────
#
# A VI iteration is `draw_samples! → update!`:
#   draw_samples! — fill the residual buffer with a Monte-Carlo set drawn at the current
#                   mean (pushforward: IID white noise; MGVI: CG solve `(I+Fisher)δ = η`;
#                   geoVI: + the nonlinear curve). The only stochastic phase.
#   update!       — estimate the KL with that fixed set and move the variational parameters
#                   (one `Optimisers` step, or `NewtonCG` to convergence).
# This is the `draw → optimize-against-fixed-samples → resample` loop (NIFTy's
# `OptimizeVI.update`); a custom loop can call the two phases directly.

"""
    draw_samples!(rng, problem, state) -> state

VI phase 1: fill `state.residuals` with a fresh Monte-Carlo sample set drawn at the current
parameters, dispatching to the family's `GeoVI.draw_samples!`. The only stochastic phase.
No-op for a MAP problem (no samples). Unexported.
"""
function draw_samples!(rng::AbstractRNG, problem::VariationalProblem, state::VIState)
    state.residuals === nothing && return state
    draw_samples!(
        problem.family, problem.likelihood, state.position, state.residuals, rng,
        problem.estimator.mirrored,
    )
    return state
end

"""
    update!(problem, state) -> state

VI phase 2: estimate the KL with the current samples and move the variational
mean (the position optimization), writing it back into `state.position`.
Unexported.
"""
function update!(problem::VariationalProblem, state::VIState)
    result = _optimize_position(problem, state.position, state.residuals, state.optimizer_state)
    # `fmap(copyto!, …)` writes leaf-wise into the existing parameter buffers
    # (preserving their identity for Reactant in-place aliasing); for a bare-array
    # θ this is exactly `copyto!(state.position, result.x)`.
    fmap(copyto!, state.position, result.x)
    state.optimizer_state = result.optimizer_state
    return state
end

# ── Objective: family × divergence ─────────────────────────────────────────

_negative_logposterior(lh::AbstractLikelihood, x::AbstractArray) =
    -logdensity(lh, x) + 0.5 * real(dot(x, x))

# The one reverse-KL objective for every family (Fisher-Gaussian, mean-field, and any
# pushforward family): `mean_i[-log p(ξ_i) + log q_θ(ξ_i)]`. Fisher-Gaussian families
# take the `logdensity` default 0 and recover `mean_i[-log p(ξ_i)]`.
function _fdivergence_value(
        family::AbstractVariationalFamily,
        ::ReverseKL,
        lh::AbstractLikelihood,
        position,
        residuals,
    )
    # MAP (no samples) is only reachable for a bare-array family (θ is the latent point);
    # structured-θ families require `n_samples > 0`.
    residuals === nothing && return _negative_logposterior(lh, position)

    value = zero(eltype(residuals))
    n = _sample_count(residuals)
    # `@trace for` so the n-fold Monte Carlo sum compiles to a single MLIR while-loop
    # body instead of n trace-time-unrolled iterations. `value` is already a
    # `TracedRNumber` here (via `zero(eltype(...))`), so no explicit promotion is needed.
    @trace track_numbers = false for i in 1:n
        r = _sample_slice(residuals, i)
        # `transport_and_logjac` returns both the latent sample ξ and the reparameterization's
        # log-Jacobian; the reverse-KL objective is `mean_i[-log p(ξ_i) - logjac_i]`.
        ξ, logjac = transport_and_logjac(family, position, r)
        value = value + _negative_logposterior(lh, ξ) - logjac
    end
    return value / n
end

function _fdivergence_value(
        ::AbstractVariationalFamily,
        ::ForwardKL,
        lh::AbstractLikelihood,
        position::AbstractArray,
        residuals,
    )
    throw(
        ArgumentError(
            "`ForwardKL` is not implemented yet; only `ReverseKL()` is supported in `fit`",
        ),
    )
end

# ── AD plumbing ────────────────────────────────────────────────────────────

function _finite_difference_value_and_gradient(
        objective,
        x::AbstractArray;
        relstep::Real = 1.0e-6,
    )
    relstep > 0 || throw(ArgumentError("`relstep` must be positive"))

    value = objective(x)
    grad = similar(x, eltype(x))
    xp = copy(x)
    xm = copy(x)

    for I in eachindex(x)
        xi = x[I]
        step = relstep * max(1.0, abs(float(real(xi))))
        xp[I] = xi + step
        xm[I] = xi - step
        grad[I] = (objective(xp) - objective(xm)) / (2 * step)
        xp[I] = xi
        xm[I] = xi
    end

    return value, grad
end

# Tree-structured θ (e.g. mean-field's NamedTuple): flatten with `destructure`,
# run the array finite-difference on the flat vector, and reconstruct the gradient
# into the same structure. Arrays dispatch to the `::AbstractArray` method above,
# so this only fires for non-array parameter containers.
function _finite_difference_value_and_gradient(objective, x; relstep::Real = 1.0e-6)
    flat, re = Optimisers.destructure(x)
    value, gflat = _finite_difference_value_and_gradient(objective ∘ re, flat; relstep = relstep)
    return value, re(gflat)
end

function _unsupported_adtype_message(adtype)
    return "AD choice $(typeof(adtype)) is not available. Load the corresponding AD package/extension or choose a supported `ADTypes` backend."
end

function _value_and_gradient(
        ::ADTypes.AutoFiniteDiff,
        objective,
        x;
        fd_eps::Real = 1.0e-6,
    )
    return _finite_difference_value_and_gradient(objective, x; relstep = fd_eps)
end

function _value_and_gradient(
        ::ADTypes.NoAutoDiff,
        objective,
        x;
        fd_eps::Real = 1.0e-6,
    )
    throw(
        ArgumentError(
            "`NoAutoDiff()` disables differentiation; choose a concrete AD backend like `AutoFiniteDiff()` or `AutoEnzyme()`",
        ),
    )
end

function _value_and_gradient(
        adtype::ADTypes.AbstractADType,
        objective,
        x;
        fd_eps::Real = 1.0e-6,
    )
    throw(ArgumentError(_unsupported_adtype_message(adtype)))
end

# ── Outer position update ──────────────────────────────────────────────────

function _require_supported(
        family::AbstractVariationalFamily,
        divergence::AbstractFDivergence,
        estimator::AbstractEstimator,
        optimizer,
    )
    divergence isa ReverseKL || throw(
        ArgumentError(
            "`fit` currently supports `ReverseKL()` only; got $(typeof(divergence))",
        ),
    )
    # `NewtonCG` is natural-gradient: it needs the family's metric. Any family may opt in
    # via `supports_natural_gradient`; the Fisher-Gaussian families do, mean-field (and
    # other pushforward families) do not — they take a bare `Optimisers.jl` rule.
    if optimizer isa NewtonCG
        supports_natural_gradient(family) || throw(
            ArgumentError(
                "`$(nameof(typeof(family)))` does not support `NewtonCG` (no natural-gradient " *
                    "metric); use an `Optimisers.jl` rule (e.g. `Optimisers.Adam`).",
            ),
        )
    elseif !(optimizer isa Optimisers.AbstractRule)
        throw(
            ArgumentError(
                "`fit` expects `NewtonCG()` or an `Optimisers.jl` rule; got $(typeof(optimizer))",
            ),
        )
    end
    # Mean-field's ELBO entropy estimate needs at least one Monte-Carlo sample.
    if family isa MeanFieldGaussian && estimator.n_samples == 0
        throw(
            ArgumentError("`MeanFieldGaussian` requires `n_samples > 0` for the ELBO estimate"),
        )
    end
    return nothing
end

_outer_vi_objective(family, divergence, likelihood, residuals, x) =
    _fdivergence_value(family, divergence, likelihood, x, residuals)

function _outer_vi_value_and_gradient(
        adtype,
        family,
        divergence,
        likelihood,
        residuals,
        x;
        fd_eps = 1.0e-6,
    )
    objective = y -> _outer_vi_objective(family, divergence, likelihood, residuals, y)
    return _value_and_gradient(adtype, objective, x; fd_eps = fd_eps)
end

function _optimize_position(problem::VariationalProblem, position, residuals, opt_state)
    optimizer = problem.optimizer
    family = problem.family
    divergence = problem.divergence
    likelihood = problem.likelihood
    adtype = problem.adtype
    return _optimize(
        optimizer,
        position;
        fun_and_grad = x -> _outer_vi_value_and_gradient(
            adtype,
            family,
            divergence,
            likelihood,
            residuals,
            x,
        ),
        # The metric closure is built unconditionally but invoked only by `NewtonCG`; a
        # bare `Optimisers.jl` rule ignores it (so a family without a metric is fine there).
        metricp = (x, v) -> natural_gradient_metric(family, likelihood, x, residuals, v),
        optimizer_state = opt_state,
        _optimizer_kwargs(optimizer)...,
    )
end

# ── Step / fit ─────────────────────────────────────────────────────────────

"""
    step_vi!(rng, problem, state::VIState) -> state

Advance the VI loop in place, reusing `state` and its buffers: a fresh
[`draw_samples!`](@ref GeoVI.draw_samples!) → [`update!`](@ref GeoVI.update!) cycle —
draw a Monte-Carlo set at the current mean, then optimize the variational parameters
against that fixed set. `rng` is advanced in place; `problem` is the immutable config.

`step_vi!` is a pure in-place mutation with no host-only state, so under Reactant
you compile it yourself and call the compiled thunk in your loop:

```julia
rng, state = init(rng, problem)
cstep = @compile step_vi!(rng, problem, state)
for _ in 1:n; cstep(rng, problem, state); end
```

[`fit`](@ref) does this for you. For finer control, compose the phase primitives
`GeoVI.draw_samples!` / `GeoVI.update!` directly.
"""
function step_vi!(rng, problem::VariationalProblem, state::VIState)
    draw_samples!(rng, problem, state)
    update!(problem, state)
    return state
end

# Drive `n_iterations` of `step_vi!`. The eager path loops directly; the Reactant
# extension overrides this to `@compile step_vi!` once and loop the compiled thunk.
function _run_vi!(::Any, rng, problem::VariationalProblem, state::VIState, n_iterations)
    for _ in 1:n_iterations
        step_vi!(rng, problem, state)
    end
    return state
end

"""
    fit([rng], problem, n_iterations) -> AbstractVariationalDistribution

Convenience driver: run `n_iterations` of [`step_vi!`](@ref) and return the fitted
variational distribution ([`distribution`](@ref)). Under Reactant it compiles `step_vi!`
once and loops the compiled thunk. `rng` defaults to `Random.default_rng()`.
"""
function fit(rng::AbstractRNG, problem::VariationalProblem, n_iterations::Integer)
    n_iterations >= 0 || throw(ArgumentError("`n_iterations` must be non-negative"))
    rng, state = init(rng, problem)
    _run_vi!(problem.adtype, rng, problem, state, n_iterations)
    return distribution(problem, state)
end

fit(problem::VariationalProblem, n_iterations::Integer; rng::AbstractRNG = Random.default_rng()) =
    fit(rng, problem, n_iterations)

"""
    distribution(problem, state::VIState) -> AbstractVariationalDistribution

The fitted variational distribution `q_θ` at the current state — `distribution(family, θ,
likelihood)` with `θ = state.position`. It is a pure distribution (no retained samples):
draw from it with `rand(rng, q[, n])`, and `logdensity(q, ξ)` where the family supports it.
"""
distribution(problem::VariationalProblem, state::VIState) =
    distribution(problem.family, state.position, problem.likelihood)

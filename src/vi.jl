"""
    AbstractFDivergence

Marker supertype for f-divergence objectives optimized by the outer VI loop.

The objective hook `_fdivergence_value(family, divergence, ...)` dispatches jointly on
the family and the divergence, so a new scheme may ship its own objective form.
"""
abstract type AbstractFDivergence end
struct ReverseKL <: AbstractFDivergence end
struct ForwardKL <: AbstractFDivergence end

# The problem input is a *position* only. `step_vi!` redraws the residuals at the start
# of every step, so initial residuals could never influence a fit — accepting them would
# only invite `update!`-before-`draw_samples!` misuse.
function _problem_samples(samples::Samples)
    samples.residuals === nothing || throw(
        ArgumentError(
            "the initial `Samples` must not carry residuals — `step_vi!` redraws them " *
                "at the start of every step, so they would never be used; pass the " *
                "position alone",
        ),
    )
    return samples
end
_problem_samples(position::AbstractArray) = Samples(position, nothing; keys = nothing)

"""
    VariationalProblem(lh, position_or_samples; family, divergence, estimator, optimizer, adtype)

Bundle a likelihood with the four orthogonal VI axes (`family`, `divergence`,
`estimator`, `optimizer`) plus the AD backend. `position_or_samples` is the
initial latent point `ξ₀` (a flat white array) or a [`Samples`](@ref) whose
`position` is `ξ₀` (and whose residuals must be `nothing` — [`step_vi!`](@ref)
redraws the sample set at the start of every step, so initial residuals could
never be used).

Every construction path validates the axis combination (`NewtonCG` requires a
natural-gradient family; structured-θ families require `n_samples > 0`; only
`ReverseKL` is implemented).

!!! note "AD backend cost"
    The default `adtype = AutoFiniteDiff()` needs no extra packages but costs
    `2·length(ξ₀)` objective evaluations per gradient — each an `n_samples`
    Monte-Carlo sum over the forward model. For anything beyond toy problems
    load Enzyme and pass `adtype = AutoEnzyme()` (inferred as `AutoReactant`
    automatically for Reactant arrays).
"""
struct VariationalProblem{L, S, F, D, E, O, AD}
    likelihood::L
    initial_samples::S
    family::F
    divergence::D
    estimator::E
    optimizer::O
    adtype::AD

    function VariationalProblem(lh, position_or_samples, family, divergence, estimator, optimizer, adtype)
        samples = _problem_samples(position_or_samples)
        # Validation is host-only: under a Reactant trace the problem may be
        # reconstructed with traced fields, where re-validating is wasted work.
        within_compile() ||
            _require_supported(family, divergence, estimator, optimizer, samples.position)
        return new{
            typeof(lh), typeof(samples), typeof(family), typeof(divergence),
            typeof(estimator), typeof(optimizer), typeof(adtype),
        }(lh, samples, family, divergence, estimator, optimizer, adtype)
    end
end

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

# Zero-filled, never uninitialized: there is no "residuals not yet drawn" state after
# `init`. (Zero residuals are the degenerate sample set collapsed onto the mean, so even a
# custom loop calling `GeoVI.update!` before the first `GeoVI.draw_samples!` is
# well-defined, not reading garbage.)
function _init_residual_buffer(problem::VariationalProblem, latent)
    n_stored = _n_stored_samples(problem.estimator)
    n_stored == 0 && return nothing
    residuals = similar(latent, (n_stored, size(latent)...))
    return fill!(residuals, zero(eltype(residuals)))
end

# Fresh optimizer state for the chosen optimizer (`nothing` for the stateless
# `NewtonCG`, an `Optimisers` setup for a rule). Threaded across steps.
_init_optimizer_state(problem::VariationalProblem, position) =
    _optimizer_state(problem.optimizer, position, nothing)

"""
    init([rng], problem[, ξ0]) -> (rng, state)

Allocate the [`VIState`](@ref) for `problem` — a fresh copy of the initial
position, the residual buffer, and the optimizer state — and return it together
with the loop RNG to thread through [`step_vi!`](@ref).

`ξ0` optionally overrides the problem's initial latent point
(`problem.initial_samples.position`); it is the **starting latent point** (for
MGVI/geoVI the mean the optimizer starts from, for mean-field the seed for
`(; mean = ξ0, logstd = 0)`). This is how you restart from a new point without
rebuilding the `problem`. To restart an *existing* state in place (reusing its
buffers), use [`reset!`](@ref) instead.

`rng` is first-positional or auto-generated (`Random.default_rng()`); it is never
a keyword. For an `AutoReactant` problem the RNG is wrapped into a
`Reactant.ReactantRNG` (the compiled step is built lazily by `fit`, or by the
user calling `@compile step_vi!(...)`).

The residual buffer starts zero-filled; [`step_vi!`](@ref) redraws it at the
start of every step.
"""
function init(rng::AbstractRNG, problem::VariationalProblem, ξ0 = nothing)
    # All buffers are sized from the latent point ξ₀ (latent-shaped by
    # construction), so the interface needs no `θ → latent` projection.
    latent = something(ξ0, problem.initial_samples.position)
    θ = init_params(problem.family, latent)
    residuals = _init_residual_buffer(problem, latent)
    optimizer_state = _init_optimizer_state(problem, θ)
    wrapped_rng = _wrap_rng(problem.adtype, rng)
    return wrapped_rng, VIState(θ, residuals, optimizer_state)
end

init(problem::VariationalProblem) = init(Random.default_rng(), problem)
init(problem::VariationalProblem, ξ0) = init(Random.default_rng(), problem, ξ0)

# Latent-sized reference array inside a θ container, for reset!'s size check.
# Array families: θ *is* the latent. Mean-field: θ.mean is latent-sized.
_latent_ref(θ::AbstractArray) = θ
_latent_ref(θ) = first(values(θ))

"""
    reset!(state, problem, ξ0) -> state

Re-initialize an existing [`VIState`](@ref) in place to restart from a new
starting latent point `ξ0`, reusing `state`'s buffers instead of allocating.
This is [`init`](@ref) for an already-allocated state: the variational parameters
are rebuilt from `ξ0` via `init_params`, the residual buffer is zeroed (it is
redrawn at the start of every [`step_vi!`](@ref) anyway), and the optimizer state
is reset to fresh (momentum cleared).

`state.position` and `state.residuals` keep their array identity (so a compiled
`step_vi!` thunk stays valid under Reactant); the optimizer state is reassigned,
matching [`update!`](@ref)'s per-step behavior. `ξ0` must match the existing
latent size — `reset!` cannot resize; call [`init`](@ref) for a different size.
"""
function reset!(state::VIState, problem::VariationalProblem, ξ0::AbstractArray)
    ref = _latent_ref(state.position)
    size(ref) == size(ξ0) || throw(
        DimensionMismatch(
            "reset! cannot resize: state latent size $(size(ref)), new ξ0 size " *
                "$(size(ξ0)). Use `init` to allocate a fresh state of the new size."
        ),
    )
    θ_new = init_params(problem.family, ξ0)
    # Leaf-wise in-place copy preserves position-buffer identity (cf. `update!`);
    # for a bare-array θ this is exactly `copyto!(state.position, θ_new)`.
    fmap(copyto!, state.position, θ_new)
    state.residuals === nothing ||
        fill!(state.residuals, zero(eltype(state.residuals)))
    # Reassign (do NOT fmap-copy): optimizer trees have non-array leaves.
    state.optimizer_state = _init_optimizer_state(problem, state.position)
    return state
end

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
        estimator::AbstractEstimator,
        lh::AbstractLikelihood,
        position::AbstractArray,
        rng::AbstractRNG,
    )
    n = _n_base_draws(estimator)
    mirrored = _mirrored(estimator)
    if n == 0
        return Samples(position, nothing; keys = nothing),
            (family = family, mirrored = mirrored, n_draws = 0)
    end

    n_stored = _n_stored_samples(estimator)
    residuals = similar(position, (n_stored, size(position)...))
    draw_samples!(family, lh, position, residuals, rng, mirrored)

    # `keys` indexes the STORED rows (mirrored pairs count as two), matching
    # `length(samples)`.
    return Samples(position, residuals; keys = Base.OneTo(n_stored)),
        (family = family, mirrored = mirrored, n_draws = n)
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
        _mirrored(problem.estimator),
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
        position,
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
    # With no samples the objective degenerates to the negative log-posterior at θ
    # (MAP), which is only defined when θ is itself the latent point. A structured-θ
    # family (NamedTuple parameters, e.g. mean-field) therefore needs samples.
    if _n_stored_samples(estimator) == 0 && position !== nothing
        θ0 = init_params(family, position)
        θ0 isa AbstractArray || throw(
            ArgumentError(
                "`$(nameof(typeof(family)))` has structured parameters " *
                    "(θ::$(typeof(θ0)) is not the latent point), so the sample-free MAP " *
                    "objective is undefined for it; use an estimator with `n_samples > 0`.",
            ),
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
        metricp = NaturalGradientField(family, likelihood, residuals),
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

fit(problem::VariationalProblem, n_iterations::Integer) =
    fit(Random.default_rng(), problem, n_iterations)

"""
    distribution(problem, state::VIState) -> AbstractVariationalDistribution

The fitted variational distribution `q_θ` at the current state — `distribution(family, θ,
likelihood)` with `θ = state.position`. It is a pure distribution (no retained samples):
draw from it with `rand(rng, q[, n])`, and `logdensity(q, ξ)` where the family supports it.
"""
distribution(problem::VariationalProblem, state::VIState) =
    distribution(problem.family, state.position, problem.likelihood)

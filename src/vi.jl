"""
    AbstractVariationalFamily

Marker supertype for variational schemes. A family defines *how a sample is
drawn from the variational distribution* `q` (and pairs with a divergence to
define the objective). It carries the solvers that the draw needs, but no
outer-optimization or Monte-Carlo-budget knobs — those live on the
[`MCEstimator`](@ref) estimator and the optimizer respectively.

To add a new scheme, subtype `AbstractVariationalFamily`, implement
`_draw_sample_block(family, lh, position, rng, mirrored)` and
`_draw_one_residual(family, lh, position, rng)`, and provide
`_fdivergence_value` / `_fdivergence_fishermetric` methods for the
`(family, divergence)` pairs it supports.
"""
abstract type AbstractVariationalFamily end

"""
    MGVIFamily(; solver=ConjugateGradient())

Metric Gaussian VI: the variational distribution is the Gaussian whose
covariance is the inverse posterior Fisher metric. A draw solves
`(I + Fisher) ξ = sample` with `solver`.
"""
struct MGVIFamily{S} <: AbstractVariationalFamily
    solver::S
end
MGVIFamily(; solver = ConjugateGradient()) = MGVIFamily(solver)

"""
    GeoVIFamily(; solver=ConjugateGradient(), curve=NewtonCG())

Geometric VI: a draw is an MGVI linear residual (`solver`) refined by a
nonlinear coordinate "curve" obtained by minimizing the geoVI residual
objective with `curve` (a [`NewtonCG`](@ref)). The curve is part of the *draw*,
not the outer fit.
"""
struct GeoVIFamily{S, C} <: AbstractVariationalFamily
    solver::S
    curve::C
end
GeoVIFamily(; solver = ConjugateGradient(), curve = NewtonCG()) = GeoVIFamily(solver, curve)

"""
    AbstractFDivergence

Marker supertype for f-divergence objectives optimized by the outer VI loop.

The objective hooks `_fdivergence_value(family, divergence, ...)` and
`_fdivergence_fishermetric(family, divergence, ...)` dispatch jointly on the
family and the divergence, so a new scheme may ship its own objective.
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
    _require_supported(divergence, optimizer)
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
once by [`init`](@ref) with every field at its final type — the current mean
(`position`), the residual buffer (`residuals`), the stored per-base-draw white
noise (`metric_white`, `prior_white`) that [`transform!`](@ref GeoVI.transform!)
de-whitens (and that custom loops reuse to recompute the transform at a moved
mean), and the threaded `optimizer_state` — advanced in place, so reusing one
`VIState` keeps per-iteration allocation flat. It holds no host-only fields (no
iteration counter, no compile cache), so [`step_vi!`](@ref) is a pure in-place
mutation the user can `@compile` directly under Reactant.
"""
mutable struct VIState{P, R, M, Pw, Os}
    position::P
    residuals::R
    metric_white::M
    prior_white::Pw
    optimizer_state::Os
end

_wrap_rng(_adtype, rng) = rng

function _init_residual_buffer(problem::VariationalProblem, position)
    problem.estimator.n_samples == 0 && return nothing
    return similar(position, (problem.estimator.n_samples, size(position)...))
end

# Per-base-draw white-noise buffers (one row per base draw): `metric_white` in
# the likelihood tangent space, `prior_white` in latent space. `sample!` fills
# them; `transform!` de-whitens them. Stored so a custom loop can replay a draw
# at a moved mean (recompute the Fisher). `nothing` when there are no samples (MAP).
function _init_white_buffers(problem::VariationalProblem, position)
    n = _n_base_draws(problem.estimator)
    n == 0 && return nothing, nothing
    tangent = _tangent_template(problem.adtype, problem.likelihood, position)
    metric_white = similar(tangent, (n, size(tangent)...))
    prior_white = similar(position, (n, size(position)...))
    return metric_white, prior_white
end

# Fresh optimizer state for the chosen optimizer (`nothing` for the stateless
# `NewtonCG`, an `Optimisers` setup for a rule). Threaded across steps.
_init_optimizer_state(problem::VariationalProblem, position) =
    _optimizer_state(problem.optimizer, position, nothing)

"""
    init([rng], problem) -> (rng, state)

Allocate the [`VIState`](@ref) for `problem` — a fresh copy of the initial
position, the residual buffer, the white-noise buffers, and the optimizer state —
and return it together with the loop RNG to thread through [`step_vi!`](@ref). For
an `AutoReactant` problem the RNG is wrapped into a `Reactant.ReactantRNG` (the
compiled step is built lazily by `fit`, or by the user calling
`@compile step_vi!(...)`). `rng` defaults to `Random.default_rng()`.
"""
function init(rng::AbstractRNG, problem::VariationalProblem)
    position = copy(problem.initial_samples.position)
    residuals = _init_residual_buffer(problem, position)
    metric_white, prior_white = _init_white_buffers(problem, position)
    optimizer_state = _init_optimizer_state(problem, position)
    wrapped_rng = _wrap_rng(problem.adtype, rng)
    return wrapped_rng,
        VIState(position, residuals, metric_white, prior_white, optimizer_state)
end

init(problem::VariationalProblem; rng::AbstractRNG = Random.default_rng()) =
    init(rng, problem)

# ── Sample-block plumbing (Reactant-safe) ──────────────────────────────────

function _single_sample_block(residual::AbstractArray)
    return reshape(residual, (1, size(residual)...))
end

_sample_block_size(block::AbstractArray) = size(block, 1)

function _allocate_sample_residuals(block::AbstractArray, n_blocks)
    block_size = size(block, 1)
    trailing_dims = ntuple(i -> size(block, i + 1), max(ndims(block) - 1, 0))
    return similar(block, (block_size * n_blocks, trailing_dims...))
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

# ── Per-family draws ───────────────────────────────────────────────────────

function _draw_sample_block(
        family::MGVIFamily,
        lh::AbstractLikelihood,
        position::AbstractArray,
        rng::AbstractRNG,
        mirrored::Bool,
    )
    linear_draw = draw_linear_residual(
        lh, position, rng; _draw_linear_kwargs(family.solver)...
    )
    block = mirrored ?
        _stack_residuals(linear_draw.residual, -linear_draw.residual) :
        _single_sample_block(linear_draw.residual)
    return block, linear_draw
end

function _draw_sample_block(
        family::GeoVIFamily,
        lh::AbstractLikelihood,
        position::AbstractArray,
        rng::AbstractRNG,
        mirrored::Bool,
    )
    curve_options = _optimizer_kwargs(family.curve)
    linear_draw = draw_linear_residual(
        lh, position, rng; _draw_linear_kwargs(family.solver)...
    )

    if mirrored
        positive_update = update_nonlinear_residual(
            lh,
            position,
            linear_draw;
            optimizer = family.curve,
            optimizer_options = curve_options,
        )
        negative_update = update_nonlinear_residual(
            lh,
            position,
            -linear_draw.residual;
            metric_sample = linear_draw.metric_sample,
            metric_sample_sign = -1,
            optimizer = family.curve,
            optimizer_options = curve_options,
        )
        draw = MirroredResidualDraw(
            _stack_residuals(positive_update.residual, negative_update.residual),
            linear_draw,
            positive_update,
            negative_update,
        )
        return draw.residuals, draw
    end

    curved_update = update_nonlinear_residual(
        lh,
        position,
        linear_draw;
        optimizer = family.curve,
        optimizer_options = curve_options,
    )
    return _single_sample_block(curved_update.residual), curved_update
end

"""
    _draw_one_residual(family, lh, position, rng)

Draw a single residual from the variational distribution (no mirroring).
Used by `rand` on a [`VariationalPosterior`](@ref).
"""
function _draw_one_residual(
        family::MGVIFamily, lh::AbstractLikelihood, position::AbstractArray, rng::AbstractRNG
    )
    return draw_linear_residual(
        lh, position, rng; _draw_linear_kwargs(family.solver)...
    ).residual
end

function _draw_one_residual(
        family::GeoVIFamily, lh::AbstractLikelihood, position::AbstractArray, rng::AbstractRNG
    )
    linear_draw = draw_linear_residual(
        lh, position, rng; _draw_linear_kwargs(family.solver)...
    )
    update = update_nonlinear_residual(
        lh,
        position,
        linear_draw;
        optimizer = family.curve,
        optimizer_options = _optimizer_kwargs(family.curve),
    )
    return update.residual
end

"""
    draw_residuals(problem, position, rng)
    draw_residuals(family, estimator, lh, position, rng)

Draw the Monte-Carlo residual set used to estimate `E_q[·]`: `estimator`
controls the count/mirroring, `family` controls how each draw is realized.
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

    first_block, _ = _draw_sample_block(family, lh, position, rng, estimator.mirrored)
    residuals = _allocate_sample_residuals(first_block, n)
    _write_sample_block!(residuals, 1, first_block)

    # `@trace for` so the n-fold sample draws compile to a single MLIR
    # while-loop body instead of (n - 1) unrolled copies of the full
    # CG + forward model graph. `track_numbers = false` keeps the deep
    # type-walk away from plain Int/Bool fields in the closure environment.
    @trace track_numbers = false for i in 2:n
        block, _ = _draw_sample_block(family, lh, position, rng, estimator.mirrored)
        _write_sample_block!(residuals, i, block)
    end

    return Samples(position, residuals; keys = Base.OneTo(n)),
        (family = family, mirrored = estimator.mirrored, n_draws = n)
end

# ── The three VI phases (composable primitives; unexported) ─────────────────
#
# A VI iteration is the reparameterization structure of VI, sliced into phases:
#   sample!    — draw white noise ξ_w ∼ N(0, I) from the reference distribution.
#   transform! — de-whiten: apply the family transform at the current mean (MGVI:
#                CG solve `(I+Fisher)δ = η`; geoVI: + the nonlinear curve) to turn
#                the stored noise into posterior-sample residuals. Re-running it
#                with the same stored noise recomputes the Fisher at the new mean.
#   update!    — estimate the KL with the samples and move the variational mean.
# `step_vi!` runs sample! → transform! → update!; users can compose them directly
# (e.g. `sample!` once then `transform!`+`update!` to refine at a fixed noise).

# Phase 1 (array level): fill the white-noise buffers from N(0, I). No-op for a
# MAP problem (no samples → `nothing` buffers).
_draw_white_noise!(::Nothing, ::Nothing, ::AbstractRNG) = nothing
function _draw_white_noise!(metric_white, prior_white, rng::AbstractRNG)
    copyto!(metric_white, randn_like(rng, metric_white))
    copyto!(prior_white, randn_like(rng, prior_white))
    return nothing
end

# Phase 2 (one base draw): de-whiten the stored noise into a residual block
# (1 row, or 2 antithetic rows when mirrored). The CG (and geoVI curve) live here.
# The geoVI curve runs with `throw_on_failure = false` (cf. NIFTy.re's
# `_raise_notconverged = False`): re-transforming at a converged mean can leave
# the curve unable to improve an already-optimal residual, which is not an error.
function _transform_block(family::MGVIFamily, lh, position, ms, mirrored)
    linear = draw_linear_residual(lh, position, ms; _draw_linear_kwargs(family.solver)...)
    return mirrored ?
        _stack_residuals(linear.residual, -linear.residual) :
        _single_sample_block(linear.residual)
end

function _transform_block(family::GeoVIFamily, lh, position, ms, mirrored)
    curve_options = _optimizer_kwargs(family.curve)
    linear = draw_linear_residual(lh, position, ms; _draw_linear_kwargs(family.solver)...)
    pos = update_nonlinear_residual(
        lh, position, linear;
        optimizer = family.curve, optimizer_options = curve_options, throw_on_failure = false,
    )
    mirrored || return _single_sample_block(pos.residual)
    neg = update_nonlinear_residual(
        lh, position, -linear.residual;
        metric_sample = ms, metric_sample_sign = -1,
        optimizer = family.curve, optimizer_options = curve_options, throw_on_failure = false,
    )
    return _stack_residuals(pos.residual, neg.residual)
end

# Phase 2 (array level): de-whiten every base draw's stored noise into `residuals`.
# No-op for a MAP problem (no samples → `nothing` residuals).
_transform!(::VariationalProblem, position, ::Nothing, metric_white, prior_white) = nothing
function _transform!(problem::VariationalProblem, position, residuals, metric_white, prior_white)
    family = problem.family
    lh = problem.likelihood
    mirrored = problem.estimator.mirrored
    n = _n_base_draws(problem.estimator)
    # `@trace for` (track_numbers = false) → one MLIR loop body, not n unrolled
    # copies of the full CG + forward-model graph.
    @trace track_numbers = false for i in 1:n
        mw = _sample_slice(metric_white, i)
        pw = _sample_slice(prior_white, i)
        ms = _metric_sample_from_white(lh, position, mw, pw)
        _write_sample_block!(residuals, i, _transform_block(family, lh, position, ms, mirrored))
    end
    return residuals
end

"""
    sample!(rng, problem, state) -> state

VI phase 1: draw fresh white noise `ξ_w ∼ N(0, I)` into the state buffers. The
only stochastic phase. No-op for a MAP problem (no samples). Unexported.
"""
function sample!(rng::AbstractRNG, problem::VariationalProblem, state::VIState)
    state.metric_white === nothing && return state
    _draw_white_noise!(state.metric_white, state.prior_white, rng)
    return state
end

"""
    transform!(problem, state) -> state

VI phase 2: de-whiten the stored noise into sample residuals at the current mean
(MGVI: CG solve; geoVI: CG + curve). Re-running it (without `sample!`) recomputes
the Fisher/transform at the moved mean for the same realization. Unexported.
"""
function transform!(problem::VariationalProblem, state::VIState)
    state.residuals === nothing && return state
    _transform!(problem, state.position, state.residuals, state.metric_white, state.prior_white)
    return state
end

"""
    update!(problem, state) -> state

VI phase 3: estimate the KL with the current samples and move the variational
mean (the position optimization), writing it back into `state.position`.
Unexported.
"""
function update!(problem::VariationalProblem, state::VIState)
    result = _optimize_position(problem, state.position, state.residuals, state.optimizer_state)
    copyto!(state.position, result.x)
    state.optimizer_state = result.optimizer_state
    return state
end

# ── Objective: family × divergence ─────────────────────────────────────────

_negative_logposterior(lh::AbstractLikelihood, x::AbstractArray) =
    -logdensity(lh, x) + 0.5 * real(dot(x, x))

function _sample_position(position::AbstractArray, residuals::AbstractArray, i)
    return _sample_slice(residuals, i) .+ position
end

function _fdivergence_value(
        family::AbstractVariationalFamily,
        ::ReverseKL,
        lh::AbstractLikelihood,
        position::AbstractArray,
        residuals,
    )
    residuals === nothing && return _negative_logposterior(lh, position)

    value = zero(eltype(position))
    n = _sample_count(residuals)
    # `@trace for` so the n-fold Monte Carlo sum compiles to a single MLIR
    # while-loop body instead of n trace-time-unrolled iterations. `value`
    # is already a `TracedRNumber` here (via `zero(eltype(position))`), so no
    # explicit promotion is needed.
    @trace track_numbers = false for i in 1:n
        value = value + _negative_logposterior(lh, _sample_position(position, residuals, i))
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

function _fdivergence_fishermetric(
        family::AbstractVariationalFamily,
        ::ReverseKL,
        lh::AbstractLikelihood,
        position::AbstractArray,
        residuals,
        v::AbstractArray,
    )
    residuals === nothing && return _posterior_metric(lh, position, v)

    result = zero(v)
    n = _sample_count(residuals)
    @trace track_numbers = false for i in 1:n
        result = result .+ _posterior_metric(
            lh,
            _sample_position(position, residuals, i),
            v,
        ) ./ n
    end
    return result
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

function _unsupported_adtype_message(adtype)
    return "AD choice $(typeof(adtype)) is not available. Load the corresponding AD package/extension or choose a supported `ADTypes` backend."
end

function _value_and_gradient(
        ::ADTypes.AutoFiniteDiff,
        objective,
        x::AbstractArray;
        fd_eps::Real = 1.0e-6,
    )
    return _finite_difference_value_and_gradient(objective, x; relstep = fd_eps)
end

function _value_and_gradient(
        ::ADTypes.NoAutoDiff,
        objective,
        x::AbstractArray;
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
        x::AbstractArray;
        fd_eps::Real = 1.0e-6,
    )
    throw(ArgumentError(_unsupported_adtype_message(adtype)))
end

# ── Outer position update ──────────────────────────────────────────────────

function _require_supported(divergence::AbstractFDivergence, optimizer)
    divergence isa ReverseKL || throw(
        ArgumentError(
            "`fit` currently supports `ReverseKL()` only; got $(typeof(divergence))",
        ),
    )
    optimizer isa NewtonCG && return nothing
    optimizer isa Optimisers.AbstractRule && return nothing
    throw(
        ArgumentError(
            "`fit` expects `NewtonCG()` or an `Optimisers.jl` rule; got $(typeof(optimizer))",
        ),
    )
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

_outer_vi_metric(family, divergence, likelihood, residuals, x, v) =
    _fdivergence_fishermetric(family, divergence, likelihood, x, residuals, v)

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
        metricp = (x, v) -> _outer_vi_metric(family, divergence, likelihood, residuals, x, v),
        optimizer_state = opt_state,
        _optimizer_kwargs(optimizer)...,
    )
end

# ── Step / fit ─────────────────────────────────────────────────────────────

"""
    step_vi!(rng, problem, state::VIState, n_refine = 0) -> state

Advance the VI loop in place, reusing `state` and its buffers: a fresh
`sample!` → `transform!` → `update!` cycle, then `n_refine` extra
`transform!` → `update!` refinements that **reuse the drawn noise** — recomputing
the Fisher / re-curving at the moved mean (common random numbers, the cheap
geoVI inner loop). `rng` is advanced in place; `problem` is the immutable config.

`step_vi!` is a pure in-place mutation with no host-only state, so under Reactant
you compile it yourself and call the compiled thunk in your loop:

```julia
rng, state = init(rng, problem)
cstep = @compile step_vi!(rng, problem, state, Reactant.ConcreteRNumber(k))
for _ in 1:n; cstep(rng, problem, state, Reactant.ConcreteRNumber(k)); end
```

Passing `n_refine` as a `ConcreteRNumber{Int}` keeps it a *runtime* loop bound, so
one compiled graph serves any count (and the whole cycle fuses). [`fit`](@ref)
does this for you. For finer control, compose the phase primitives
`GeoVI.sample!` / `GeoVI.transform!` / `GeoVI.update!` directly.
"""
function step_vi!(rng, problem::VariationalProblem, state::VIState, n_refine = 0)
    sample!(rng, problem, state)
    transform!(problem, state)
    update!(problem, state)
    # `n_refine` extra refinements reusing the drawn noise. `@trace for` with a
    # `ConcreteRNumber` bound is a runtime loop (recompile-free); eager / `Int`
    # bound it is a plain loop (empty when `n_refine == 0`).
    @trace track_numbers = false for _ in 1:n_refine
        transform!(problem, state)
        update!(problem, state)
    end
    return state
end

# Drive `n_iterations` of `step_vi!`. The eager path loops directly; the Reactant
# extension overrides this to `@compile step_vi!` once (wrapping `n_refine` as a
# `ConcreteRNumber`) and loop the compiled thunk.
function _run_vi!(::Any, rng, problem::VariationalProblem, state::VIState, n_iterations, n_refine)
    for _ in 1:n_iterations
        step_vi!(rng, problem, state, n_refine)
    end
    return state
end

"""
    fit([rng], problem, n_iterations; n_refine = 0) -> VariationalPosterior

Convenience driver: run `n_iterations` of [`step_vi!`](@ref) (each with `n_refine`
noise-reusing refinements) and return the fitted [`VariationalPosterior`](@ref).
Under Reactant it compiles `step_vi!` once and loops the compiled thunk. `rng`
defaults to `Random.default_rng()`.
"""
function fit(
        rng::AbstractRNG, problem::VariationalProblem, n_iterations::Integer;
        n_refine::Integer = 0,
    )
    n_iterations >= 0 || throw(ArgumentError("`n_iterations` must be non-negative"))
    n_refine >= 0 || throw(ArgumentError("`n_refine` must be non-negative"))
    rng, state = init(rng, problem)
    _run_vi!(problem.adtype, rng, problem, state, n_iterations, n_refine)
    return posterior(problem, state)
end

fit(problem::VariationalProblem, n_iterations::Integer; rng::AbstractRNG = Random.default_rng(), n_refine::Integer = 0) =
    fit(rng, problem, n_iterations; n_refine = n_refine)

"""
    posterior(problem, state::VIState) -> VariationalPosterior

Build the fitted [`VariationalPosterior`](@ref) from `problem` and the current
`state`.
"""
function posterior(problem::VariationalProblem, state::VIState)
    keys = state.residuals === nothing ? nothing :
        Base.OneTo(_n_base_draws(problem.estimator))
    samples = Samples(state.position, state.residuals; keys = keys)
    return VariationalPosterior(
        problem.likelihood,
        state.position,
        problem.family,
        samples,
    )
end

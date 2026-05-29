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
once by [`init`](@ref) with every field at its final type — the iteration
counter, the current mean (`position`), the residual buffer (`residuals`), the
threaded `optimizer_state`, and the compilation `cache` — and advanced in place,
so reusing one `VIState` keeps per-iteration allocation flat.
"""
mutable struct VIState{P, R, Os, C}
    iteration::Int
    position::P
    residuals::R
    optimizer_state::Os
    cache::C
end

_wrap_rng(_adtype, rng) = rng

function _init_residual_buffer(problem::VariationalProblem, position)
    problem.estimator.n_samples == 0 && return nothing
    return similar(position, (problem.estimator.n_samples, size(position)...))
end

# Fresh optimizer state for the chosen optimizer (`nothing` for the stateless
# `NewtonCG`, an `Optimisers` setup for a rule). Threaded across steps.
_init_optimizer_state(problem::VariationalProblem, position) =
    _optimizer_state(problem.optimizer, position, nothing)

# Compilation cache, built once at `init`. Eager backends need none (`nothing`);
# the Reactant extension overrides this to `@compile` the step for the exact
# preallocated buffers and return a fully-populated, concretely-typed cache.
_init_cache(_adtype, problem, position, residuals, rng, optimizer_state) = nothing

"""
    init([rng], problem) -> (rng, state)

Allocate the [`VIState`](@ref) for `problem` — a fresh copy of the initial
position, the residual buffer, the optimizer state, and the compilation cache —
and return it together with the loop RNG to thread through [`step_vi!`](@ref).
Everything is built here at its final type: for an `AutoReactant` problem the
RNG is wrapped into a `Reactant.ReactantRNG` and the step is compiled for the
preallocated buffers, so the cache holds the concrete compiled step (no
recompilation on the first iteration). `rng` defaults to `Random.default_rng()`.
"""
function init(rng::AbstractRNG, problem::VariationalProblem)
    position = copy(problem.initial_samples.position)
    residuals = _init_residual_buffer(problem, position)
    optimizer_state = _init_optimizer_state(problem, position)
    wrapped_rng = _wrap_rng(problem.adtype, rng)
    cache = _init_cache(
        problem.adtype, problem, position, residuals, wrapped_rng, optimizer_state
    )
    return wrapped_rng, VIState(0, position, residuals, optimizer_state, cache)
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

# In-place residual draw used by the stepping kernel: fills the preallocated
# `residuals` buffer (or does nothing when there are no samples) and returns the
# sample-state metadata.
_draw_residuals!(::Nothing, problem::VariationalProblem, position, rng) =
    (family = problem.family, mirrored = problem.estimator.mirrored, n_draws = 0)

function _draw_residuals!(
        residuals::AbstractArray,
        problem::VariationalProblem,
        position::AbstractArray,
        rng::AbstractRNG,
    )
    family = problem.family
    estimator = problem.estimator
    lh = problem.likelihood
    n = _n_base_draws(estimator)

    first_block, _ = _draw_sample_block(family, lh, position, rng, estimator.mirrored)
    _write_sample_block!(residuals, 1, first_block)

    @trace track_numbers = false for i in 2:n
        block, _ = _draw_sample_block(family, lh, position, rng, estimator.mirrored)
        _write_sample_block!(residuals, i, block)
    end
    return (family = family, mirrored = estimator.mirrored, n_draws = n)
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
    _vi_step!(problem, position, residuals, rng, opt_state) -> OptimizationResult

One VI iteration done in place: draw residuals into `residuals`, optimize the
mean, and write it back into `position`. `rng` is advanced in place. This is the
unit compiled by the Reactant extension (Reactant traces the in-place mutation)
and run directly on the eager path. Returns the `OptimizationResult` (the caller
keeps only its `optimizer_state` for threading).
"""
function _vi_step!(problem::VariationalProblem, position, residuals, rng, opt_state)
    _draw_residuals!(residuals, problem, position, rng)
    result = _optimize_position(problem, position, residuals, opt_state)
    copyto!(position, result.x)
    return result
end

function _step_vi!(::Any, rng, problem::VariationalProblem, state::VIState)
    result = _vi_step!(
        problem, state.position, state.residuals, rng, state.optimizer_state
    )
    state.optimizer_state = result.optimizer_state
    state.iteration += 1
    return state
end

"""
    step_vi!(rng, problem, state::VIState) -> state

Advance the VI loop one iteration in place, reusing `state` and its buffers.
`rng` is advanced in place; `problem` is the immutable configuration.
"""
function step_vi!(rng, problem::VariationalProblem, state::VIState)
    _step_vi!(problem.adtype, rng, problem, state)
    return state
end

"""
    fit([rng], problem, n_iterations) -> VariationalPosterior

Convenience driver: run `n_iterations` of [`step_vi!`](@ref) and return the
fitted [`VariationalPosterior`](@ref). `rng` defaults to `Random.default_rng()`.
"""
function fit(rng::AbstractRNG, problem::VariationalProblem, n_iterations::Integer)
    n_iterations >= 0 || throw(ArgumentError("`n_iterations` must be non-negative"))
    rng, state = init(rng, problem)
    for _ in 1:n_iterations
        step_vi!(rng, problem, state)
    end
    return posterior(problem, state)
end

fit(problem::VariationalProblem, n_iterations::Integer; rng::AbstractRNG = Random.default_rng()) =
    fit(rng, problem, n_iterations)

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

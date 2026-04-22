"""
    AbstractVariationalFamily

Marker supertype for variational-sampling schemes.
"""
abstract type AbstractVariationalFamily end
struct MGVIFamily <: AbstractVariationalFamily end
struct GeoVIFamily <: AbstractVariationalFamily end

"""
    AbstractFDivergence

Marker supertype for f-divergence objectives optimized by the outer VI loop.

To add a new divergence, subtype `AbstractFDivergence` and implement the
internal objective hooks `_fdivergence_value(::YourDivergence, ...)` and
`_fdivergence_fishermetric(::YourDivergence, ...)`.
"""
abstract type AbstractFDivergence end
struct ReverseKL <: AbstractFDivergence end
struct ForwardKL <: AbstractFDivergence end

struct VIConfig{AD,DL,NU,OO}
    adtype::AD
    n_iterations::Int
    n_samples::Int
    mirrored::Bool
    draw_linear::DL
    nonlinear_update::NU
    optimizer_options::OO
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
    n_base_draws::Int
end

struct VIState{R,S,M}
    iteration::Int
    rng::R
    sample_state::S
    minimization_state::M
    cache::Any
end

function VIConfig(;
    adtype=ADTypes.AutoFiniteDiff(),
    n_iterations::Integer=0,
    n_samples::Integer=0,
    mirrored::Bool=true,
    draw_linear=(;),
    nonlinear_update=(;),
    optimizer_options=(;),
)
    adtype === nothing && (adtype = ADTypes.AutoFiniteDiff())
    n_iterations >= 0 || throw(ArgumentError("`n_iterations` must be non-negative"))
    n_samples >= 0 || throw(ArgumentError("`n_samples` must be non-negative"))
    mirrored && isodd(n_samples) &&
        throw(ArgumentError("mirrored sampling requires an even `n_samples`"))
    return VIConfig(
        adtype,
        Int(n_iterations),
        Int(n_samples),
        mirrored,
        draw_linear,
        nonlinear_update,
        optimizer_options,
    )
end

function VIState(;
    iteration::Integer=0,
    rng=nothing,
    sample_state=nothing,
    minimization_state=nothing,
    cache=nothing,
)
    return VIState(Int(iteration), rng, sample_state, minimization_state, cache)
end

function _resolve_draw_linear_options(options)
    return (
        cg_rtol=_option(options, :cg_rtol, 1e-8),
        cg_atol=_option(options, :cg_atol, 0.0),
        cg_maxiter=_option(options, :cg_maxiter, nothing),
        cg_miniter=_option(options, :cg_miniter, 0),
        throw_on_failure=_option(options, :throw_on_failure, true),
    )
end

function _resolve_nonlinear_update_options(options)
    return (
        maxiter=_option(options, :maxiter, 20),
        miniter=_option(options, :miniter, 0),
        xtol=_option(options, :xtol, 1e-5),
        absdelta=_option(options, :absdelta, nothing),
        cg_rtol=_option(options, :cg_rtol, 1e-8),
        cg_atol=_option(options, :cg_atol, 0.0),
        cg_maxiter=_option(options, :cg_maxiter, nothing),
        cg_miniter=_option(options, :cg_miniter, 0),
        throw_on_failure=_option(options, :throw_on_failure, true),
    )
end

function _resolve_optimizer_options(options)
    return (
        maxiter=_option(options, :maxiter, 20),
        miniter=_option(options, :miniter, 0),
        xtol=_option(options, :xtol, 1e-5),
        absdelta=_option(options, :absdelta, nothing),
        cg_rtol=_option(options, :cg_rtol, 1e-8),
        cg_atol=_option(options, :cg_atol, 0.0),
        cg_maxiter=_option(options, :cg_maxiter, nothing),
        cg_miniter=_option(options, :cg_miniter, 0),
        fd_eps=_option(options, :fd_eps, 1e-6),
    )
end

_n_base_draws(config::VIConfig) = config.mirrored ? (config.n_samples ÷ 2) : config.n_samples

_problem_samples(samples::Samples) = samples
_problem_samples(position::AbstractArray) = Samples(position, nothing; keys=nothing)

function _problem_adtype(config::VIConfig, samples::Samples)
    samples.position === nothing && return config.adtype
    return _infer_adtype(config.adtype, samples.position)
end

function VariationalProblem(
    lh::AbstractLikelihood,
    position_or_samples;
    family::AbstractVariationalFamily=GeoVIFamily(),
    divergence::AbstractFDivergence=ReverseKL(),
    optimizer=NewtonCG(),
    config::VIConfig=VIConfig(),
)
    samples = _problem_samples(position_or_samples)
    _require_supported(divergence, optimizer)
    return VariationalProblem(
        lh,
        samples,
        family,
        divergence,
        optimizer,
        config,
        _problem_adtype(config, samples),
        _resolve_draw_linear_options(config.draw_linear),
        _resolve_nonlinear_update_options(config.nonlinear_update),
        _resolve_optimizer_options(config.optimizer_options),
        _n_base_draws(config),
    )
end

initialize_vi(rng; config::VIConfig) = VIState(iteration=0, rng=rng)
initialize_vi(problem::VariationalProblem, rng::AbstractRNG=Random.default_rng()) =
    VIState(iteration=0, rng=rng)

function _single_sample_block(residual::AbstractArray)
    return reshape(residual, (1, size(residual)...))
end

_sample_block_size(block::AbstractArray) = size(block, 1)

function _allocate_sample_residuals(block::AbstractArray, n_blocks::Integer)
    trailing_dims = ntuple(i -> size(block, i + 1), max(ndims(block) - 1, 0))
    return similar(block, (n_blocks * _sample_block_size(block), trailing_dims...))
end

function _write_sample_block!(dest::AbstractArray, i::Integer, block::AbstractArray)
    block_size = _sample_block_size(block)
    first = (Int(i) - 1) * block_size + 1
    last = first + block_size - 1
    trailing = ntuple(_ -> Colon(), max(ndims(block) - 1, 0))
    dest[first:last, trailing...] = block
    return dest
end

_strip_cache(state::VIState) = VIState(
    iteration=state.iteration,
    rng=state.rng,
    sample_state=state.sample_state,
    minimization_state=state.minimization_state,
)

function _restore_cache(state::VIState, cache)
    return VIState(
        iteration=state.iteration,
        rng=state.rng,
        sample_state=state.sample_state,
        minimization_state=state.minimization_state,
        cache=cache,
    )
end

function _draw_sample_block(
    problem::VariationalProblem,
    ::MGVIFamily,
    position::AbstractArray,
    rng::AbstractRNG,
)
    linear_draw = draw_linear_residual(
        problem.likelihood,
        position,
        rng;
        problem.draw_linear_options...,
    )
    block = problem.config.mirrored ?
        _stack_residuals(linear_draw.residual, -linear_draw.residual) :
        _single_sample_block(linear_draw.residual)
    return block, linear_draw
end

function _draw_sample_block(
    problem::VariationalProblem,
    ::GeoVIFamily,
    position::AbstractArray,
    rng::AbstractRNG,
)
    linear_draw = draw_linear_residual(
        problem.likelihood,
        position,
        rng;
        problem.draw_linear_options...,
    )

    if problem.config.mirrored
        positive_update = update_nonlinear_residual(
            problem.likelihood,
            position,
            linear_draw;
            optimizer=problem.optimizer,
            optimizer_options=problem.nonlinear_update_options,
            throw_on_failure=problem.nonlinear_update_options.throw_on_failure,
        )
        negative_update = update_nonlinear_residual(
            problem.likelihood,
            position,
            -linear_draw.residual;
            metric_sample=linear_draw.metric_sample,
            metric_sample_sign=-1,
            optimizer=problem.optimizer,
            optimizer_options=problem.nonlinear_update_options,
            throw_on_failure=problem.nonlinear_update_options.throw_on_failure,
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
        problem.likelihood,
        position,
        linear_draw;
        optimizer=problem.optimizer,
        optimizer_options=problem.nonlinear_update_options,
        throw_on_failure=problem.nonlinear_update_options.throw_on_failure,
    )
    return _single_sample_block(curved_update.residual), curved_update
end

function _draw_samples(
    problem::VariationalProblem,
    position::AbstractArray,
    rng::AbstractRNG,
)
    if problem.config.n_samples == 0
        return Samples(position, nothing; keys=nothing),
        (family=problem.family, mirrored=problem.config.mirrored, n_draws=0)
    end

    first_block, _ = _draw_sample_block(problem, problem.family, position, rng)
    residuals = _allocate_sample_residuals(first_block, problem.n_base_draws)
    _write_sample_block!(residuals, 1, first_block)

    for i in 2:problem.n_base_draws
        block, _ = _draw_sample_block(problem, problem.family, position, rng)
        _write_sample_block!(residuals, i, block)
    end

    return Samples(position, residuals; keys=Base.OneTo(problem.n_base_draws)),
    (family=problem.family, mirrored=problem.config.mirrored, n_draws=problem.n_base_draws)
end

_negative_logposterior(lh::AbstractLikelihood, x::AbstractArray) =
    -logdensity(lh, x) + 0.5 * real(dot(x, x))

function _sample_position(position::AbstractArray, residuals::AbstractArray, i::Int)
    return _sample_slice(residuals, i) .+ position
end

function _fdivergence_value(
    ::ReverseKL,
    lh::AbstractLikelihood,
    position::AbstractArray,
    residuals,
)
    residuals === nothing && return _negative_logposterior(lh, position)

    value = zero(eltype(position))
    n = _sample_count(residuals)
    for i in 1:n
        value += _negative_logposterior(lh, _sample_position(position, residuals, i))
    end
    return value / n
end

function _fdivergence_value(
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

function _finite_difference_value_and_gradient(
    objective,
    x::AbstractArray;
    relstep::Real=1e-6,
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

_infer_adtype(adtype, x) = adtype

function _value_and_gradient(
    ::ADTypes.AutoFiniteDiff,
    objective,
    x::AbstractArray;
    fd_eps::Real=1e-6,
)
    return _finite_difference_value_and_gradient(objective, x; relstep=fd_eps)
end

function _value_and_gradient(
    ::ADTypes.NoAutoDiff,
    objective,
    x::AbstractArray;
    fd_eps::Real=1e-6,
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
    fd_eps::Real=1e-6,
)
    throw(ArgumentError(_unsupported_adtype_message(adtype)))
end

function _fdivergence_value_and_gradient(
    adtype,
    divergence::AbstractFDivergence,
    lh::AbstractLikelihood,
    position::AbstractArray,
    residuals;
    fd_eps::Real=1e-6,
)
    objective = x -> _fdivergence_value(divergence, lh, x, residuals)
    return _value_and_gradient(adtype, objective, position; fd_eps=fd_eps)
end

function _fdivergence_fishermetric(
    ::AbstractFDivergence,
    lh::AbstractLikelihood,
    position::AbstractArray,
    residuals,
    v::AbstractArray,
)
    residuals === nothing && return _posterior_metric(lh, position, v)

    result = zero(v)
    n = _sample_count(residuals)
    for i in 1:n
        result = result .+ _posterior_metric(
            lh,
            _sample_position(position, residuals, i),
            v,
        ) ./ n
    end
    return result
end

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
            "`fit` expects either `NewtonCG()` or an `Optimisers.jl` rule; got $(typeof(optimizer))",
        ),
    )
end

function _previous_optimizer_state(optimizer, x0, previous_result)
    return _optimizer_state(optimizer, x0, previous_result)
end

_outer_vi_objective(divergence, likelihood, residuals, x) =
    _fdivergence_value(divergence, likelihood, x, residuals)

function _outer_vi_value_and_gradient(
    adtype,
    divergence,
    likelihood,
    residuals,
    x;
    fd_eps=1e-6,
)
    objective = y -> _outer_vi_objective(divergence, likelihood, residuals, y)
    return _value_and_gradient(adtype, objective, x; fd_eps=fd_eps)
end

_outer_vi_metric(divergence, likelihood, residuals, x, v) =
    _fdivergence_fishermetric(divergence, likelihood, x, residuals, v)

_materialize_step_position(x) = x

function _update_position(
    problem::VariationalProblem,
    samples::Samples,
    previous_minimization_state=nothing,
)
    optimizer_state = _previous_optimizer_state(
        problem.optimizer,
        samples.position,
        previous_minimization_state,
    )
    divergence = problem.divergence
    likelihood = problem.likelihood
    residuals = samples.residuals
    adtype = problem.adtype
    fd_eps = problem.optimizer_options.fd_eps
    result = _optimize(
        problem.optimizer,
        samples.position;
        fun_and_grad=x -> _outer_vi_value_and_gradient(
            adtype,
            divergence,
            likelihood,
            residuals,
            x;
            fd_eps=fd_eps,
        ),
        hessp=(x, v) -> _outer_vi_metric(divergence, likelihood, residuals, x, v),
        maxiter=problem.optimizer_options.maxiter,
        miniter=problem.optimizer_options.miniter,
        xtol=problem.optimizer_options.xtol,
        absdelta=problem.optimizer_options.absdelta,
        cg_rtol=problem.optimizer_options.cg_rtol,
        cg_atol=problem.optimizer_options.cg_atol,
        cg_maxiter=problem.optimizer_options.cg_maxiter,
        cg_miniter=problem.optimizer_options.cg_miniter,
        optimizer_state=optimizer_state,
    )
    return result
end

function _step_vi_impl(
    problem::VariationalProblem,
    samples::Samples,
    state::VIState,
)
    drawn_samples, sample_state = _draw_samples(
        problem,
        samples.position,
        state.rng,
    )
    minimization_state = _update_position(
        problem,
        drawn_samples,
        state.minimization_state,
    )

    new_samples = Samples(
        _materialize_step_position(minimization_state.x),
        drawn_samples.residuals;
        keys=drawn_samples.keys,
    )
    new_state = VIState(
        iteration=state.iteration + 1,
        rng=state.rng,
        sample_state=sample_state,
        minimization_state=minimization_state,
    )
    return new_samples, new_state
end

function _step_vi_default(
    problem::VariationalProblem,
    samples::Samples,
    state::VIState,
)
    new_samples, new_state = _step_vi_impl(
        problem,
        samples,
        _strip_cache(state),
    )
    return new_samples, _restore_cache(new_state, state.cache)
end

function _step_vi(
    adtype,
    problem::VariationalProblem,
    samples::Samples,
    state::VIState,
)
    return _step_vi_default(problem, samples, state)
end

function step_vi(
    problem::VariationalProblem,
    position::AbstractArray,
    state::VIState,
)
    return step_vi(problem, Samples(position, nothing; keys=nothing), state)
end

function step_vi(
    problem::VariationalProblem,
    samples::Samples,
    state::VIState,
)
    return _step_vi(problem.adtype, problem, samples, state)
end

step_vi(problem::VariationalProblem, state::VIState) = step_vi(problem, problem.initial_samples, state)

function step_vi(
    lh::AbstractLikelihood,
    position::AbstractArray,
    family::AbstractVariationalFamily,
    divergence::AbstractFDivergence,
    optimizer,
    state::VIState,
    config::VIConfig,
)
    problem = VariationalProblem(
        lh,
        position;
        family=family,
        divergence=divergence,
        optimizer=optimizer,
        config=config,
    )
    return step_vi(problem, position, state)
end

function step_vi(
    lh::AbstractLikelihood,
    samples::Samples,
    family::AbstractVariationalFamily,
    divergence::AbstractFDivergence,
    optimizer,
    state::VIState,
    config::VIConfig,
)
    problem = VariationalProblem(
        lh,
        samples;
        family=family,
        divergence=divergence,
        optimizer=optimizer,
        config=config,
    )
    return step_vi(problem, samples, state)
end

function fit(
    problem::VariationalProblem;
    rng=Random.default_rng(),
)
    state = initialize_vi(problem, rng)
    samples = problem.initial_samples
    for _ in 1:problem.config.n_iterations
        samples, state = step_vi(problem, samples, state)
    end
    return samples, state
end

fit(rng::AbstractRNG, problem::VariationalProblem) = fit(problem; rng=rng)

function fit(
    lh::AbstractLikelihood,
    position_or_samples,
    family::AbstractVariationalFamily,
    divergence::AbstractFDivergence,
    optimizer;
    config::VIConfig=VIConfig(),
    rng=Random.default_rng(),
)
    problem = VariationalProblem(
        lh,
        position_or_samples;
        family=family,
        divergence=divergence,
        optimizer=optimizer,
        config=config,
    )
    return fit(problem; rng=rng)
end

function fit(
    rng::AbstractRNG,
    lh::AbstractLikelihood,
    position_or_samples,
    family::AbstractVariationalFamily,
    divergence::AbstractFDivergence,
    optimizer;
    config::VIConfig=VIConfig(),
)
    return fit(
        lh,
        position_or_samples,
        family,
        divergence,
        optimizer;
        config=config,
        rng=rng,
    )
end

function fit(
    lh::AbstractLikelihood,
    position_or_samples;
    family::AbstractVariationalFamily=GeoVIFamily(),
    divergence::AbstractFDivergence=ReverseKL(),
    optimizer=NewtonCG(),
    config::VIConfig=VIConfig(),
    rng=Random.default_rng(),
)
    return fit(lh, position_or_samples, family, divergence, optimizer; config=config, rng=rng)
end

function fit(
    rng::AbstractRNG,
    lh::AbstractLikelihood,
    position_or_samples;
    family::AbstractVariationalFamily=GeoVIFamily(),
    divergence::AbstractFDivergence=ReverseKL(),
    optimizer=NewtonCG(),
    config::VIConfig=VIConfig(),
)
    return fit(lh, position_or_samples, family, divergence, optimizer; config=config, rng=rng)
end

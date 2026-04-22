"""
    AbstractVariationalFamily

Marker supertype for variational-sampling schemes.

To add a new family, subtype `AbstractVariationalFamily` and provide a
`_draw_sample_block(::YourFamily, optimizer, likelihood, position, rng, config)`
method.
"""
abstract type AbstractVariationalFamily end
struct MGVIFamily <: AbstractVariationalFamily end
struct GeoVIFamily <: AbstractVariationalFamily end

const AbstractVIScheme = AbstractVariationalFamily
const MGVIScheme = MGVIFamily
const GeoVIScheme = GeoVIFamily

"""
    AbstractDivergence
    AbstractFDivergence

Marker supertype for divergence objectives optimized by the outer VI loop.

To add a new divergence, subtype `AbstractDivergence` and implement the
internal objective hooks `_kl_value(::YourDivergence, ...)` and
`_kl_metric(::YourDivergence, ...)`.
"""
abstract type AbstractDivergence end
const AbstractFDivergence = AbstractDivergence
struct ReverseKL <: AbstractDivergence end
struct ForwardKL <: AbstractDivergence end

"""
    AbstractOptimizer

Marker supertype for built-in optimizer backends.

To add a new optimizer backend, subtype `AbstractOptimizer` and implement
`_optimize(optimizer, x0; fun_and_grad, hessp, kwargs...)`. If the backend
needs persistent state across VI iterations, also implement
`_optimizer_state(optimizer, x0, previous_result)`.

Any `Optimisers.AbstractRule` is also accepted directly without subtyping
`AbstractOptimizer`.
"""
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

function VIConfig(;
    adtype=ADTypes.AutoFiniteDiff(),
    n_iterations::Integer=0,
    n_samples::Integer=0,
    mirrored::Bool=true,
    draw_linear=(;),
    nonlinear_update=(;),
    optimizer_options=(;),
    kl_minimize=nothing,
)
    if kl_minimize !== nothing
        optimizer_options == (;) ||
            throw(ArgumentError("specify only one of `optimizer_options` or `kl_minimize`"))
        optimizer_options = kl_minimize
    end
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

function Base.getproperty(cfg::VIConfig, name::Symbol)
    if name === :kl_minimize
        return getfield(cfg, :optimizer_options)
    end
    return getfield(cfg, name)
end

function Base.propertynames(cfg::VIConfig, private::Bool=false)
    names = fieldnames(typeof(cfg))
    return private ? names : (names..., :kl_minimize)
end

struct VIState{R,S,M}
    iteration::Int
    rng::R
    sample_state::S
    minimization_state::M
    cache::Any
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

initialize_vi(rng; config::VIConfig) = VIState(iteration=0, rng=rng)

_n_base_draws(config::VIConfig) = config.mirrored ? (config.n_samples ÷ 2) : config.n_samples

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
    ::MGVIFamily,
    optimizer,
    lh::AbstractLikelihood,
    position::AbstractArray,
    rng::AbstractRNG,
    config::VIConfig,
)
    residual, info = draw_linear_residual(lh, position, rng; config.draw_linear...)
    block = config.mirrored ? _stack_residuals(residual, -residual) : _single_sample_block(residual)
    return block, info
end

function _draw_sample_block(
    ::GeoVIFamily,
    optimizer,
    lh::AbstractLikelihood,
    position::AbstractArray,
    rng::AbstractRNG,
    config::VIConfig,
)
    if config.mirrored
        return draw_residual(
            lh,
            position,
            rng;
            draw_linear_kwargs=config.draw_linear,
            optimizer=optimizer,
            optimizer_options=config.nonlinear_update,
        )
    end

    metric_sample, prior_sample = _draw_metric_sample(lh, position, rng)
    linear_residual, linear_info = draw_linear_residual(
        lh,
        position,
        rng;
        config.draw_linear...,
        metric_sample=metric_sample,
        x0=prior_sample,
    )
    curved_residual, curved_info = update_nonlinear_residual(
        lh,
        position,
        linear_residual;
        metric_sample=metric_sample,
        optimizer=optimizer,
        optimizer_options=config.nonlinear_update,
    )
    return _single_sample_block(curved_residual), (linear=linear_info, positive=curved_info)
end

function _draw_samples(
    lh::AbstractLikelihood,
    position::AbstractArray,
    family::AbstractVariationalFamily,
    optimizer,
    rng::AbstractRNG,
    config::VIConfig,
)
    if config.n_samples == 0
        return Samples(position, nothing; keys=nothing),
        (family=family, mirrored=config.mirrored, n_draws=0)
    end

    n_draws = _n_base_draws(config)
    first_block, _ = _draw_sample_block(family, optimizer, lh, position, rng, config)
    residuals = _allocate_sample_residuals(first_block, n_draws)
    _write_sample_block!(residuals, 1, first_block)

    for i in 2:n_draws
        block, _ = _draw_sample_block(family, optimizer, lh, position, rng, config)
        _write_sample_block!(residuals, i, block)
    end

    return Samples(position, residuals; keys=Base.OneTo(n_draws)),
    (family=family, mirrored=config.mirrored, n_draws=n_draws)
end

_negative_logposterior(lh::AbstractLikelihood, x::AbstractArray) =
    -logdensity(lh, x) + 0.5 * real(dot(x, x))

function _sample_position(position::AbstractArray, residuals::AbstractArray, i::Int)
    return _sample_slice(residuals, i) .+ position
end

function _kl_value(::ReverseKL, lh::AbstractLikelihood, position::AbstractArray, residuals)
    residuals === nothing && return _negative_logposterior(lh, position)

    value = zero(eltype(position))
    n = _sample_count(residuals)
    for i in 1:n
        value += _negative_logposterior(lh, _sample_position(position, residuals, i))
    end
    return value / n
end

function _kl_value(::ForwardKL, lh::AbstractLikelihood, position::AbstractArray, residuals)
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

function _kl_value_and_gradient(
    adtype,
    divergence::AbstractDivergence,
    lh::AbstractLikelihood,
    position::AbstractArray,
    residuals;
    fd_eps::Real=1e-6,
)
    objective = x -> _kl_value(divergence, lh, x, residuals)
    return _value_and_gradient(adtype, objective, position; fd_eps=fd_eps)
end

function _kl_metric(
    ::AbstractDivergence,
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

function _require_supported(divergence::AbstractDivergence, optimizer)
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

function _update_position(
    lh::AbstractLikelihood,
    divergence::AbstractDivergence,
    optimizer,
    samples::Samples,
    config::VIConfig,
    previous_minimization_state=nothing,
)
    _require_supported(divergence, optimizer)

    fd_eps = _option(config.optimizer_options, :fd_eps, 1e-6)
    adtype = _infer_adtype(config.adtype, samples.position)
    optimizer_state = _previous_optimizer_state(
        optimizer,
        samples.position,
        previous_minimization_state,
    )
    objective = x -> _kl_value(divergence, lh, x, samples.residuals)
    result = _optimize(
        optimizer,
        samples.position;
        fun_and_grad=x -> _value_and_gradient(adtype, objective, x; fd_eps=fd_eps),
        hessp=(x, v) -> _kl_metric(divergence, lh, x, samples.residuals, v),
        maxiter=_option(config.optimizer_options, :maxiter, 20),
        miniter=_option(config.optimizer_options, :miniter, 0),
        xtol=_option(config.optimizer_options, :xtol, 1e-5),
        absdelta=_option(config.optimizer_options, :absdelta, nothing),
        cg_rtol=_option(config.optimizer_options, :cg_rtol, 1e-8),
        cg_atol=_option(config.optimizer_options, :cg_atol, 0.0),
        cg_maxiter=_option(config.optimizer_options, :cg_maxiter, nothing),
        cg_miniter=_option(config.optimizer_options, :cg_miniter, 0),
        optimizer_state=optimizer_state,
    )
    return result
end

function _step_vi_impl(
    lh::AbstractLikelihood,
    samples::Samples,
    family::AbstractVariationalFamily,
    divergence::AbstractDivergence,
    optimizer,
    state::VIState,
    config::VIConfig,
)
    drawn_samples, sample_state = _draw_samples(
        lh,
        samples.position,
        family,
        optimizer,
        state.rng,
        config,
    )
    minimization_state = _update_position(
        lh,
        divergence,
        optimizer,
        drawn_samples,
        config,
        state.minimization_state,
    )

    new_samples = Samples(minimization_state.x, drawn_samples.residuals; keys=drawn_samples.keys)
    new_state = VIState(
        iteration=state.iteration + 1,
        rng=state.rng,
        sample_state=sample_state,
        minimization_state=minimization_state,
    )
    return new_samples, new_state
end

function _step_vi_default(
    adtype,
    lh::AbstractLikelihood,
    samples::Samples,
    family::AbstractVariationalFamily,
    divergence::AbstractDivergence,
    optimizer,
    state::VIState,
    config::VIConfig,
)
    new_samples, new_state = _step_vi_impl(
        lh,
        samples,
        family,
        divergence,
        optimizer,
        _strip_cache(state),
        config,
    )
    return new_samples, _restore_cache(new_state, state.cache)
end

function _step_vi(
    adtype,
    lh::AbstractLikelihood,
    samples::Samples,
    family::AbstractVariationalFamily,
    divergence::AbstractDivergence,
    optimizer,
    state::VIState,
    config::VIConfig,
)
    return _step_vi_default(adtype, lh, samples, family, divergence, optimizer, state, config)
end

function step_vi(
    lh::AbstractLikelihood,
    position::AbstractArray,
    family::AbstractVariationalFamily,
    divergence::AbstractDivergence,
    optimizer,
    state::VIState,
    config::VIConfig,
)
    return step_vi(
        lh,
        Samples(position, nothing; keys=nothing),
        family,
        divergence,
        optimizer,
        state,
        config,
    )
end

function step_vi(
    lh::AbstractLikelihood,
    samples::Samples,
    family::AbstractVariationalFamily,
    divergence::AbstractDivergence,
    optimizer,
    state::VIState,
    config::VIConfig,
)
    adtype = _infer_adtype(config.adtype, samples.position)
    return _step_vi(adtype, lh, samples, family, divergence, optimizer, state, config)
end

function fit(
    lh::AbstractLikelihood,
    position::AbstractArray,
    family::AbstractVariationalFamily,
    divergence::AbstractDivergence,
    optimizer;
    config::VIConfig=VIConfig(),
    rng=Random.default_rng(),
)
    state = initialize_vi(rng; config=config)
    samples = Samples(position, nothing; keys=nothing)
    for _ in 1:config.n_iterations
        samples, state = step_vi(lh, samples, family, divergence, optimizer, state, config)
    end
    return samples, state
end

function fit(
    lh::AbstractLikelihood,
    samples::Samples,
    family::AbstractVariationalFamily,
    divergence::AbstractDivergence,
    optimizer;
    config::VIConfig=VIConfig(),
    rng=Random.default_rng(),
)
    state = initialize_vi(rng; config=config)
    current = samples
    for _ in 1:config.n_iterations
        current, state = step_vi(lh, current, family, divergence, optimizer, state, config)
    end
    return current, state
end

function fit(
    rng::AbstractRNG,
    lh::AbstractLikelihood,
    position_or_samples,
    family::AbstractVariationalFamily,
    divergence::AbstractDivergence,
    optimizer;
    config::VIConfig=VIConfig(),
)
    return fit(lh, position_or_samples, family, divergence, optimizer; config=config, rng=rng)
end

function fit(
    lh::AbstractLikelihood,
    position_or_samples;
    family::AbstractVariationalFamily=GeoVIFamily(),
    divergence::AbstractDivergence=ReverseKL(),
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
    divergence::AbstractDivergence=ReverseKL(),
    optimizer=NewtonCG(),
    config::VIConfig=VIConfig(),
)
    return fit(lh, position_or_samples, family, divergence, optimizer; config=config, rng=rng)
end

optimize_vi(args...; kwargs...) = fit(args...; kwargs...)
optimize_kl(args...; kwargs...) = fit(args...; kwargs...)

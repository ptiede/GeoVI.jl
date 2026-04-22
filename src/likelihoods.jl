"""
    AbstractLikelihood

Abstract supertype for likelihood objects defined in the observation space.

New likelihood implementations should define:

- `logdensity(lh, y)`
- `normalized_residual(lh, y)`
- `transformation(lh, y)`
- `leftsqrtmetric(lh, y, η)`
- `rightsqrtmetric(lh, y, v)`

`fishermetric(lh, y, v)` defaults to the composition of the left and right
square-root actions, and `energy(lh, y)` remains available as the compatibility
alias `-logdensity(lh, y)`.
"""
abstract type AbstractLikelihood end

"""
    logdensity(lh, y)

Return the unnormalized log density induced by `lh` at prediction `y`, up to
additive constants independent of `y`.
"""
function logdensity end

function normalized_residual end
function transformation end
function leftsqrtmetric end
function rightsqrtmetric end
function fishermetric end

(lh::AbstractLikelihood)(y) = logdensity(lh, y)

energy(lh::AbstractLikelihood, y) = -logdensity(lh, y)
metric(lh::AbstractLikelihood, y, v) = fishermetric(lh, y, v)
leftsqrtfishermetric(lh::AbstractLikelihood, y, η) = leftsqrtmetric(lh, y, η)
rightsqrtfishermetric(lh::AbstractLikelihood, y, v) = rightsqrtmetric(lh, y, v)

function fishermetric(lh::AbstractLikelihood, y, v)
    return leftsqrtmetric(lh, y, rightsqrtmetric(lh, y, v))
end

_sum_logdensity(x) = real(sum(x))

_infer_sqrt_precision(precision::Number) = sqrt(precision)
_infer_sqrt_precision(precision::UniformScaling) = sqrt(precision.λ) * I
_infer_sqrt_precision(precision::Diagonal) = Diagonal(sqrt.(diag(precision)))
function _infer_sqrt_precision(precision::AbstractArray)
    ndims(precision) == 2 &&
        size(precision, 1) == size(precision, 2) &&
        throw(
            ArgumentError(
                "matrix precision requires an explicit `sqrt_precision` operator",
            ),
        )
    return sqrt.(precision)
end
function _infer_sqrt_precision(precision)
    throw(
        ArgumentError(
            "cannot infer `sqrt_precision` from a callable precision; pass it explicitly",
        ),
    )
end

function _apply_operator(op::Number, x::AbstractArray)
    return op .* x
end
_apply_operator(op::UniformScaling, x) = op * x
_apply_operator(op::Diagonal, x::AbstractVecOrMat) = op * x
_apply_operator(op::AbstractMatrix, x::AbstractVecOrMat) = op * x
function _apply_operator(op::AbstractArray, x::AbstractArray)
    size(op) == size(x) || throw(
        DimensionMismatch("array operators must either be matrices or match the operand size"),
    )
    return op .* x
end
function _apply_operator(op, x)
    applicable(op, x) || throw(ArgumentError("operator $(typeof(op)) cannot act on $(typeof(x))"))
    return op(x)
end

function _bernoulli_transform_scalar(x::Real)
    z = x / 2
    if z > 0
        return π - 2 * atan(exp(-z))
    end
    return 2 * atan(exp(z))
end

_is_binary_value(x) = x == zero(x) || x == one(x)

function _validate_nonnegative(name::AbstractString, x)
    ok = x isa Number ? (x >= 0) : all(x .>= 0)
    ok || throw(ArgumentError("`$name` must be non-negative"))
    return x
end

function _validate_binary_data(data)
    ok = data isa Number ? _is_binary_value(data) : all(_is_binary_value, data)
    ok ||
        throw(ArgumentError("Bernoulli observations must be binary (0 or 1)"))
    return data
end

function _validate_binomial_data(data, trials)
    _validate_nonnegative("trials", trials)
    ok = data isa Number && trials isa Number ? (0 <= data <= trials) : all((0 .<= data) .& (data .<= trials))
    ok ||
        throw(ArgumentError("binomial successes must lie in `[0, trials]`"))
    return data, trials
end

struct GaussianLikelihood{D,P,S} <: AbstractLikelihood
    data::D
    precision::P
    sqrt_precision::S
end

function GaussianLikelihood(data; precision=1, sqrt_precision=nothing)
    sqrt_precision === nothing && (sqrt_precision = _infer_sqrt_precision(precision))
    return GaussianLikelihood(data, precision, sqrt_precision)
end

function logdensity(lh::GaussianLikelihood, y)
    resid = lh.data - y
    return -0.5 * real(dot(resid, _apply_operator(lh.precision, resid)))
end

normalized_residual(lh::GaussianLikelihood, y) =
    _apply_operator(lh.sqrt_precision, lh.data - y)
transformation(lh::GaussianLikelihood, y) = _apply_operator(lh.sqrt_precision, y)
leftsqrtmetric(lh::GaussianLikelihood, y, η) = _apply_operator(lh.sqrt_precision, η)
rightsqrtmetric(lh::GaussianLikelihood, y, v) = _apply_operator(lh.sqrt_precision, v)
fishermetric(lh::GaussianLikelihood, y, v) = _apply_operator(lh.precision, v)

struct PoissonLikelihood{D,W} <: AbstractLikelihood
    data::D
    weight::W
end

function PoissonLikelihood(data; weight=1)
    _validate_nonnegative("weight", weight)
    return PoissonLikelihood(data, weight)
end

_poisson_rate(y) = exp.(y)
_poisson_metric_diag(lh::PoissonLikelihood, y) = lh.weight .* _poisson_rate(y)
_poisson_sqrtmetric_diag(lh::PoissonLikelihood, y) = sqrt.(_poisson_metric_diag(lh, y))

function logdensity(lh::PoissonLikelihood, y)
    λ = _poisson_rate(y)
    return -_sum_logdensity(lh.weight .* (λ .- lh.data .* y))
end

function normalized_residual(lh::PoissonLikelihood, y)
    λ = _poisson_rate(y)
    return sqrt.(lh.weight) .* (lh.data .- λ) ./ sqrt.(λ)
end

function transformation(lh::PoissonLikelihood, y)
    return 2 .* sqrt.(lh.weight) .* exp.(0.5 .* y)
end

leftsqrtmetric(lh::PoissonLikelihood, y, η) = _poisson_sqrtmetric_diag(lh, y) .* η
rightsqrtmetric(lh::PoissonLikelihood, y, v) = _poisson_sqrtmetric_diag(lh, y) .* v
fishermetric(lh::PoissonLikelihood, y, v) = _poisson_metric_diag(lh, y) .* v

struct BernoulliLikelihood{D,W} <: AbstractLikelihood
    data::D
    weight::W
end

function BernoulliLikelihood(data; weight=1)
    _validate_binary_data(data)
    _validate_nonnegative("weight", weight)
    return BernoulliLikelihood(data, weight)
end

_bernoulli_mean(y) = logistic.(y)
function _bernoulli_metric_diag(lh::BernoulliLikelihood, y)
    p = _bernoulli_mean(y)
    return lh.weight .* p .* (1 .- p)
end
_bernoulli_sqrtmetric_diag(lh::BernoulliLikelihood, y) = sqrt.(_bernoulli_metric_diag(lh, y))

function logdensity(lh::BernoulliLikelihood, y)
    return -_sum_logdensity(lh.weight .* (log1pexp.(y) .- lh.data .* y))
end

function normalized_residual(lh::BernoulliLikelihood, y)
    p = _bernoulli_mean(y)
    return sqrt.(lh.weight) .* (lh.data .- p) ./ sqrt.(p .* (1 .- p))
end

function transformation(lh::BernoulliLikelihood, y)
    return sqrt.(lh.weight) .* _bernoulli_transform_scalar.(y)
end

leftsqrtmetric(lh::BernoulliLikelihood, y, η) = _bernoulli_sqrtmetric_diag(lh, y) .* η
rightsqrtmetric(lh::BernoulliLikelihood, y, v) = _bernoulli_sqrtmetric_diag(lh, y) .* v
fishermetric(lh::BernoulliLikelihood, y, v) = _bernoulli_metric_diag(lh, y) .* v

struct BinomialLikelihood{D,T,W} <: AbstractLikelihood
    data::D
    trials::T
    weight::W
end

function BinomialLikelihood(data; trials, weight=1)
    _validate_binomial_data(data, trials)
    _validate_nonnegative("weight", weight)
    return BinomialLikelihood(data, trials, weight)
end

_binomial_mean(lh::BinomialLikelihood, y) = lh.trials .* _bernoulli_mean(y)
function _binomial_metric_diag(lh::BinomialLikelihood, y)
    p = _bernoulli_mean(y)
    return lh.weight .* lh.trials .* p .* (1 .- p)
end
_binomial_sqrtmetric_diag(lh::BinomialLikelihood, y) = sqrt.(_binomial_metric_diag(lh, y))

function logdensity(lh::BinomialLikelihood, y)
    return -_sum_logdensity(lh.weight .* (lh.trials .* log1pexp.(y) .- lh.data .* y))
end

function normalized_residual(lh::BinomialLikelihood, y)
    μ = _binomial_mean(lh, y)
    p = _bernoulli_mean(y)
    σ = sqrt.(lh.trials .* p .* (1 .- p))
    return sqrt.(lh.weight) .* (lh.data .- μ) ./ σ
end

function transformation(lh::BinomialLikelihood, y)
    return sqrt.(lh.weight .* lh.trials) .* _bernoulli_transform_scalar.(y)
end

leftsqrtmetric(lh::BinomialLikelihood, y, η) = _binomial_sqrtmetric_diag(lh, y) .* η
rightsqrtmetric(lh::BinomialLikelihood, y, v) = _binomial_sqrtmetric_diag(lh, y) .* v
fishermetric(lh::BinomialLikelihood, y, v) = _binomial_metric_diag(lh, y) .* v

function _unsupported_composed_adtype_message(adtype)
    return "automatic linearization is not available for `$(typeof(adtype))`; pass manual `linearize` or `pushforward`/`pullback` to `compose`, or choose a supported `adtype`."
end

function _directional_fd_step(x::AbstractArray, v::AbstractArray; relstep::Real=1e-6)
    nv = norm(v)
    nv == 0 && return 0.0
    return relstep * max(1.0, norm(x)) / nv
end

function _finite_difference_pullback(
    forward,
    x::AbstractArray,
    η::AbstractArray;
    relstep::Real=1e-6,
)
    relstep > 0 || throw(ArgumentError("`relstep` must be positive"))

    objective = z -> real(dot(forward(z), η))
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

    return grad
end

_infer_composed_adtype(adtype, x) = adtype

struct _ManualLinearizer{F,PF,PB}
    forward::F
    pushforward::PF
    pullback::PB
end

struct _ManualLinearization{V,X,PF,PB}
    value::V
    x::X
    pushforward::PF
    pullback::PB
end

function _evaluate_linearizer(linearize, x)
    return linearize(x)
end

function _evaluate_linearizer(linearize::_ManualLinearizer, x)
    return _ManualLinearization(linearize.forward(x), x, linearize.pushforward, linearize.pullback)
end

struct _FiniteDifferenceLinearization{F,X,V,T}
    forward::F
    x::X
    value::V
    fd_eps::T
end

function pushforward(lin, v::AbstractArray)
    return getproperty(lin, :pushforward)(v)
end

function pullback(lin, η::AbstractArray)
    return getproperty(lin, :pullback)(η)
end

function pushforward(lin::_ManualLinearization, v::AbstractArray)
    lin.pushforward === nothing && throw(
        ArgumentError(
            "composed likelihood needs a pushforward action; pass `linearize` or `pushforward` to `compose`",
        ),
    )
    return lin.pushforward(lin.x, v)
end

function pullback(lin::_ManualLinearization, η::AbstractArray)
    lin.pullback === nothing && throw(
        ArgumentError(
            "composed likelihood needs a pullback action; pass `linearize` or `pullback` to `compose`",
        ),
    )
    return lin.pullback(lin.x, η)
end

function pushforward(lin::_FiniteDifferenceLinearization, v::AbstractArray)
    step = _directional_fd_step(lin.x, v; relstep=lin.fd_eps)
    step == 0 && return zero(lin.value)
    xp = similar(lin.x)
    xm = similar(lin.x)
    @. xp = lin.x + step * v
    @. xm = lin.x - step * v
    return (lin.forward(xp) .- lin.forward(xm)) ./ (2 * step)
end

function pullback(lin::_FiniteDifferenceLinearization, η::AbstractArray)
    return _finite_difference_pullback(lin.forward, lin.x, η; relstep=lin.fd_eps)
end

function _manual_linearize(forward, pushforward, pullback)
    return _ManualLinearizer(forward, pushforward, pullback)
end

function _automatic_linearize(
    ::ADTypes.AutoFiniteDiff,
    forward,
    x::AbstractArray;
    fd_eps::Real=1e-6,
)
    return _FiniteDifferenceLinearization(forward, x, forward(x), fd_eps)
end

function _automatic_linearize(
    ::ADTypes.NoAutoDiff,
    forward,
    x::AbstractArray;
    fd_eps::Real=1e-6,
)
    throw(
        ArgumentError(
            "`NoAutoDiff()` disables automatic linearization; pass a manual `linearize` or `pushforward`/`pullback` to `compose`, or choose a concrete AD backend.",
        ),
    )
end

function _automatic_linearize(
    adtype::ADTypes.AbstractADType,
    forward,
    x::AbstractArray;
    fd_eps::Real=1e-6,
)
    throw(ArgumentError(_unsupported_composed_adtype_message(adtype)))
end

struct ComposedLikelihood{L,F,LIN,AD,AO} <: AbstractLikelihood
    likelihood::L
    forward::F
    linearize::LIN
    adtype::AD
    autodiff_options::AO
end

function ComposedLikelihood(
    likelihood::AbstractLikelihood,
    forward;
    linearize=nothing,
    pushforward=nothing,
    pullback=nothing,
    adtype=ADTypes.AutoFiniteDiff(),
    autodiff_options=(;),
)
    if linearize !== nothing && (pushforward !== nothing || pullback !== nothing)
        throw(
            ArgumentError(
                "specify either `linearize` or `pushforward`/`pullback`, not both",
            ),
        )
    end
    if linearize === nothing && (pushforward !== nothing || pullback !== nothing)
        linearize = _manual_linearize(forward, pushforward, pullback)
    end
    return ComposedLikelihood(
        likelihood,
        forward,
        linearize,
        adtype,
        autodiff_options,
    )
end

function compose(
    lh::AbstractLikelihood,
    forward;
    linearize=nothing,
    pushforward=nothing,
    pullback=nothing,
    adtype=ADTypes.AutoFiniteDiff(),
    autodiff_options=(;),
)
    return ComposedLikelihood(
        lh,
        forward;
        linearize=linearize,
        pushforward=pushforward,
        pullback=pullback,
        adtype=adtype,
        autodiff_options=autodiff_options,
    )
end

function _composed_linearization(lh::ComposedLikelihood, x)
    if lh.linearize !== nothing
        return _evaluate_linearizer(lh.linearize, x)
    end
    adtype = _infer_composed_adtype(lh.adtype, x)
    return _automatic_linearize(adtype, lh.forward, x; lh.autodiff_options...)
end

logdensity(lh::ComposedLikelihood, x) = logdensity(lh.likelihood, lh.forward(x))
normalized_residual(lh::ComposedLikelihood, x) =
    normalized_residual(lh.likelihood, lh.forward(x))
transformation(lh::ComposedLikelihood, x) = transformation(lh.likelihood, lh.forward(x))

function rightsqrtmetric(lh::ComposedLikelihood, x, v)
    linearization = _composed_linearization(lh, x)
    return rightsqrtmetric(
        lh.likelihood,
        linearization.value,
        pushforward(linearization, v),
    )
end

function leftsqrtmetric(lh::ComposedLikelihood, x, η)
    linearization = _composed_linearization(lh, x)
    lifted = leftsqrtmetric(lh.likelihood, linearization.value, η)
    return pullback(linearization, lifted)
end

function fishermetric(lh::ComposedLikelihood, x, v)
    linearization = _composed_linearization(lh, x)
    lifted = fishermetric(
        lh.likelihood,
        linearization.value,
        pushforward(linearization, v),
    )
    return pullback(linearization, lifted)
end

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
    ok = data isa Number && trials isa Number ? (0 <= data <= trials) :
         all((0 .<= data) .& (data .<= trials))
    ok ||
        throw(ArgumentError("binomial successes must lie in `[0, trials]`"))
    return data, trials
end

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

include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")
include("likelihoods/bernoulli.jl")
include("likelihoods/binomial.jl")
include("likelihoods/composed.jl")

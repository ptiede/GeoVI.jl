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

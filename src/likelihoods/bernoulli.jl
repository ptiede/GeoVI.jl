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

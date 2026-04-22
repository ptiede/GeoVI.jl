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

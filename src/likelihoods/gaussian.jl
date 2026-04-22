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

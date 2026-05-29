"""
    AbstractEstimator

Marker supertype for Monte-Carlo estimators of the divergence expectation
`E_q[·]`. The estimator owns *how many* sample nodes are used (and whether they
are antithetically mirrored) — a property of the expectation estimate, not of
the variational family or the solvers.
"""
abstract type AbstractEstimator end

"""
    MCEstimator(; n_samples=0, mirrored=true)

Plain Monte-Carlo estimator. `n_samples` is the number of stored sample nodes;
with `mirrored=true` they are drawn as antithetic pairs (requires even
`n_samples`) and the package draws `n_samples ÷ 2` base residuals.
"""
struct MCEstimator <: AbstractEstimator
    n_samples::Int
    mirrored::Bool
end

function MCEstimator(; n_samples = 0, mirrored = true)
    n_samples >= 0 || throw(ArgumentError("`n_samples` must be non-negative"))
    mirrored && isodd(n_samples) &&
        throw(ArgumentError("mirrored sampling requires an even `n_samples`"))
    return MCEstimator(Int(n_samples), mirrored)
end

_n_base_draws(e::MCEstimator) = e.mirrored ? (e.n_samples ÷ 2) : e.n_samples

_metric_tangent_template(lh::AbstractLikelihood, xi) = normalized_residual(lh, xi)
_posterior_metric(lh::AbstractLikelihood, xi, v) = fishermetric(lh, xi, v) .+ v

struct _PosteriorMetricOperator{L, X}
    lh::L
    xi::X
end
(op::_PosteriorMetricOperator)(v) = _posterior_metric(op.lh, op.xi, v)

struct MetricSample{M, P}
    metric::M
    prior::P
end

struct LinearResidualDraw{R, S, I}
    residual::R
    metric_sample::S
    info::I
end

function draw_metric_sample(lh::AbstractLikelihood, xi, rng::AbstractRNG)
    metric_white = randn_like(rng, _metric_tangent_template(lh, xi))
    likelihood_sample = leftsqrtmetric(lh, xi, metric_white)
    prior_sample = randn_like(rng, xi)
    return MetricSample(likelihood_sample .+ prior_sample, prior_sample)
end

function _resolve_metric_sample(metric_sample, prior_sample)
    metric_sample isa MetricSample && return metric_sample
    prior_sample === nothing && throw(
        ArgumentError(
            "explicit `metric_sample` inputs require the matching `prior_sample` initial guess",
        ),
    )
    return MetricSample(metric_sample, prior_sample)
end

"""
    draw_linear_residual(lh, xi, rng; kwargs...)
    draw_linear_residual(lh, xi, metric_sample; kwargs...)

Draw a linear MGVI residual around expansion point `xi`.

The draw is performed in white latent coordinates, so the covariance of the
returned residual is the inverse posterior Fisher metric

`I + fishermetric(lh, xi, ·)`.
"""
function draw_linear_residual(
        lh::AbstractLikelihood,
        xi,
        rng::AbstractRNG;
        cg_rtol::Real = 1.0e-8,
        cg_atol::Real = 0.0,
        cg_maxiter::Union{Nothing, Integer} = nothing,
        cg_miniter::Integer = 0,
        throw_on_failure::Bool = true,
    )
    metric_sample = draw_metric_sample(lh, xi, rng)
    return draw_linear_residual(
        lh,
        xi,
        metric_sample;
        cg_rtol = cg_rtol,
        cg_atol = cg_atol,
        cg_maxiter = cg_maxiter,
        cg_miniter = cg_miniter,
        throw_on_failure = throw_on_failure,
    )
end

function draw_linear_residual(
        lh::AbstractLikelihood,
        xi,
        metric_sample::MetricSample;
        cg_rtol::Real = 1.0e-8,
        cg_atol::Real = 0.0,
        cg_maxiter::Union{Nothing, Integer} = nothing,
        cg_miniter::Integer = 0,
        throw_on_failure::Bool = true,
    )
    cg = ConjugateGradient(rtol = cg_rtol, atol = cg_atol, maxiter = cg_maxiter, miniter = cg_miniter)
    residual, info = solve(cg, _PosteriorMetricOperator(lh, xi), metric_sample.metric; x0 = metric_sample.prior)

    if throw_on_failure && _runtime_failure_enabled(metric_sample.metric) && !info.converged
        throw(
            ErrorException(
                "conjugate gradient failed to converge after $(info.iterations) iterations",
            ),
        )
    end

    return LinearResidualDraw(residual, metric_sample, info)
end

function draw_linear_residual(
        lh::AbstractLikelihood,
        xi,
        metric_sample;
        prior_sample = nothing,
        cg_rtol::Real = 1.0e-8,
        cg_atol::Real = 0.0,
        cg_maxiter::Union{Nothing, Integer} = nothing,
        cg_miniter::Integer = 0,
        throw_on_failure::Bool = true,
    )
    return draw_linear_residual(
        lh,
        xi,
        _resolve_metric_sample(metric_sample, prior_sample);
        cg_rtol = cg_rtol,
        cg_atol = cg_atol,
        cg_maxiter = cg_maxiter,
        cg_miniter = cg_miniter,
        throw_on_failure = throw_on_failure,
    )
end

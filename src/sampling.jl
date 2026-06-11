"""
    AbstractEstimator

Marker supertype for Monte-Carlo estimators of the divergence expectation
`E_q[·]`. The estimator owns *how many* sample nodes are used (and whether they
are antithetically mirrored) — a property of the expectation estimate, not of
the variational family or the solvers.

A custom estimator implements `GeoVI._n_stored_samples` (rows in the residual
buffer), `GeoVI._mirrored` (whether those rows are antithetic ±pairs), and
`GeoVI._n_base_draws` (independent base draws). The VI loop consults only these
accessors, never struct fields.
"""
abstract type AbstractEstimator end

function _estimator_interface_error(e::AbstractEstimator)
    throw(
        ArgumentError(
            "`$(nameof(typeof(e)))` must implement `GeoVI._n_stored_samples`, " *
                "`GeoVI._mirrored`, and `GeoVI._n_base_draws` to be used as an estimator.",
        ),
    )
end

_n_stored_samples(e::AbstractEstimator) = _estimator_interface_error(e)
_mirrored(e::AbstractEstimator) = _estimator_interface_error(e)
_n_base_draws(e::AbstractEstimator) = _estimator_interface_error(e)

"""
    MCEstimator(; n_samples=4, mirrored=true)

Plain Monte-Carlo estimator. `n_samples` is the number of stored sample nodes;
with `mirrored=true` they are drawn as antithetic pairs (requires even
`n_samples`) and the package draws `n_samples ÷ 2` base residuals.
`n_samples = 0` requests no samples at all — the objective degenerates to the
negative log-posterior at the current parameters (MAP), which is only defined
for families whose parameters are themselves the latent point (MGVI/geoVI).
"""
struct MCEstimator <: AbstractEstimator
    n_samples::Int
    mirrored::Bool
end

function MCEstimator(; n_samples = 4, mirrored = true)
    n_samples >= 0 || throw(ArgumentError("`n_samples` must be non-negative"))
    mirrored && isodd(n_samples) &&
        throw(ArgumentError("mirrored sampling requires an even `n_samples`"))
    return MCEstimator(Int(n_samples), mirrored)
end

_n_stored_samples(e::MCEstimator) = e.n_samples
_mirrored(e::MCEstimator) = e.mirrored
_n_base_draws(e::MCEstimator) = e.mirrored ? (e.n_samples ÷ 2) : e.n_samples

# Tangent-space template (a forward-model evaluation) used to size the metric white noise.
# It is now built INSIDE `draw_samples!`, which runs inside the compiled `step_vi!` under
# Reactant, so it is traced normally — no host-side `@jit` shim is needed.
_metric_tangent_template(lh::AbstractLikelihood, xi) = normalized_residual(lh, xi)
_posterior_metric(lh::AbstractLikelihood, xi, v) = fishermetric(lh, xi, v) .+ v

# The posterior-metric operator `v ↦ (I + Fisher)v` pins the likelihood at `xi` ONCE
# (`_at_point`), so a CG solve reuses the forward-model linearization across all of its
# matvecs instead of rebuilding it per application.
struct _PosteriorMetricOperator{H}
    h::H
end
_PosteriorMetricOperator(lh::AbstractLikelihood, xi) = _PosteriorMetricOperator(_at_point(lh, xi))
(op::_PosteriorMetricOperator)(v) = fishermetric(op.h, v) .+ v

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
    prior_sample = randn_like(rng, xi)
    return _metric_sample_from_white(lh, xi, metric_white, prior_sample)
end

"""
    _metric_sample_from_white(lh, xi, metric_white, prior_white)

Assemble a [`MetricSample`](@ref) from already-drawn, position-independent white
noise: `metric_white` in the likelihood tangent space and `prior_white` in latent
space. The left square-root metric is (re-)applied at the current `xi`, so the same
white noise yields a *consistent* metric sample at any expansion point — the metric is
recomputed at the current mean each time `draw_samples!` runs.
"""
function _metric_sample_from_white(lh::AbstractLikelihood, xi, metric_white, prior_white)
    likelihood_sample = leftsqrtmetric(lh, xi, metric_white)
    return MetricSample(likelihood_sample .+ prior_white, prior_white)
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
        cg_rtol::Union{Nothing, Real} = 1.0e-8,
        cg_atol::Union{Nothing, Real} = 0.0,
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
        cg_rtol::Union{Nothing, Real} = 1.0e-8,
        cg_atol::Union{Nothing, Real} = 0.0,
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
        cg_rtol::Union{Nothing, Real} = 1.0e-8,
        cg_atol::Union{Nothing, Real} = 0.0,
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

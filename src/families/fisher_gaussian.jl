# ── Shared Fisher-Gaussian machinery ─────────────────────────────────────────
# MGVI and geoVI are the same family — a Gaussian with covariance `(I + Fisher)⁻¹`,
# sampled by a CG solve. geoVI only adds a nonlinear coordinate curve to the draw. They
# share every method except that one refinement step (`_refine_residual`, defined in
# `mgvi.jl` / `geovi.jl`), so the draw and the natural-gradient metric both dispatch on
# this union.

const FisherGaussian = Union{MGVIFamily, GeoVIFamily}

# The frozen draw: fill the residual buffer with the full Monte-Carlo set. The draw uses
# two-space white noise — `metric` in the likelihood tangent space, `prior` in latent
# space — bulk-drawn FIRST and OUTSIDE the `@trace for` (the rng never enters the traced
# loop, so this compiles under Reactant), in the order metric-leaf then prior-leaf. The
# `@trace for` over base draws then de-whitens each slice into a residual block: build the
# metric sample, CG-solve for the linear residual, refine (MGVI: none; geoVI: the curve).
# Family-level linear-draw options: the solver's CG settings plus the family's
# `strict` flag (eager CG non-convergence throws only when `strict = true`; under a
# Reactant trace runtime throwing is disabled wholesale by `_runtime_failure_enabled`).
_draw_linear_kwargs(fam::FisherGaussian) =
    (; _draw_linear_kwargs(fam.solver)..., throw_on_failure = fam.strict)

function draw_samples!(
        fam::FisherGaussian, lh::AbstractLikelihood, μ, residuals, rng::AbstractRNG,
        mirrored::Bool,
    )
    n = mirrored ? size(residuals, 1) ÷ 2 : size(residuals, 1)
    t = _metric_tangent_template(lh, μ)
    metric_white = randn_like(rng, similar(t, (n, size(t)...)))
    prior_white = randn_like(rng, similar(μ, (n, size(μ)...)))
    # `@trace for` (track_numbers = false) → one MLIR while-loop body, not n unrolled copies
    # of the full CG + forward-model graph.
    @trace track_numbers = false for i in 1:n
        ms = _metric_sample_from_white(
            lh, μ, _sample_slice(metric_white, i), _sample_slice(prior_white, i)
        )
        linear = draw_linear_residual(lh, μ, ms; _draw_linear_kwargs(fam)...)
        _write_sample_block!(residuals, i, _refine_residual(fam, lh, μ, linear, ms, mirrored))
    end
    return residuals
end

# The metric *is* the (sample-averaged) pinned covariance operator `I + Fisher`, reused as
# the natural-gradient preconditioner — which is exactly why `NewtonCG` is available for
# these families and not for the pushforward ones.
supports_natural_gradient(::FisherGaussian) = true

function natural_gradient_metric(
        family::FisherGaussian, lh::AbstractLikelihood, θ, residuals, v::AbstractArray
    )
    # `θ` is the latent point for the Fisher-Gaussian families, so it is the MAP point.
    residuals === nothing && return _posterior_metric(lh, θ, v)

    acc = zero(v)
    n = _sample_count(residuals)
    @trace track_numbers = false for i in 1:n
        acc .+= fishermetric(
            lh,
            first(transport_and_logjac(family, θ, _sample_slice(residuals, i))),
            v,
        )
    end
    # mean_i(Fisher_i v) + v — accumulate first, divide once.
    acc ./= n
    return acc .+ v
end

# Pinning the field at a base point (`NaturalGradientField` evaluation, once per Newton
# iteration): pin every sample point's likelihood handle (`_at_point`, caching the
# forward-model linearization), so each inner-CG matvec only *applies* the cached
# linearizations. Under a Reactant trace return the generic re-deriving
# `NaturalGradientOperator` instead — a host vector of per-sample handles cannot be
# indexed by a traced loop variable, and unrolling n forward models into the CG body is
# exactly what the `@trace` loops avoid. (`within_compile()` is a host-side,
# compile-time branch, never a traced one. XLA dedupes the re-derived linearizations
# inside the traced loop body anyway.)
struct _CachedFisherOperator{H}
    handles::H
    n::Int
end
function (op::_CachedFisherOperator)(v)
    acc = zero(v)
    for h in op.handles
        acc .+= fishermetric(h, v)
    end
    acc ./= op.n
    return acc .+ v
end

function _natural_gradient_operator(
        family::FisherGaussian, lh::AbstractLikelihood, base, residuals
    )
    residuals === nothing && return _PosteriorMetricOperator(lh, base)
    within_compile() && return NaturalGradientOperator(family, lh, base, residuals)
    n = _sample_count(residuals)
    handles = map(
        i -> _at_point(lh, first(transport_and_logjac(family, base, _sample_slice(residuals, i)))),
        1:n,
    )
    return _CachedFisherOperator(handles, n)
end


# ── The fitted distribution: shared by MGVI and geoVI ────────────────────────
# A draw is a CG solve against the posterior Fisher at the mean (geoVI adds the curve), so
# the distribution must carry the family (solver/curve) and the likelihood. Its `rand`
# reuses the same draw pipeline as a fit step (one base draw → `transport_and_logjac`).
# There is no normalized `logdensity(q, ξ)` (the normalization is an intractable
# log-determinant), but `logdensity_unnormalized` IS available: the metric-Gaussian
# log-density minus that constant, which is all `pareto_diagnostic` needs.

struct FisherGaussianDistribution{F <: FisherGaussian, V, L} <: AbstractVariationalDistribution
    family::F
    mean::V
    likelihood::L
end

distribution(family::FisherGaussian, θ, likelihood) = FisherGaussianDistribution(family, θ, likelihood)

function Base.rand(rng::AbstractRNG, d::FisherGaussianDistribution)
    fam, lh, μ = d.family, d.likelihood, d.mean
    ms = _metric_sample_from_white(
        lh, μ, randn_like(rng, _metric_tangent_template(lh, μ)), randn_like(rng, μ)
    )
    linear = draw_linear_residual(lh, μ, ms; _draw_linear_kwargs(fam)...)
    block = _refine_residual(fam, lh, μ, linear, ms, false)
    return first(transport_and_logjac(fam, μ, _sample_slice(block, 1)))
end

"""
    logdensity_unnormalized(d::FisherGaussianDistribution, ξ) -> Real

`log q(ξ) = -½ (ξ-μ)ᵀ (I+F(μ)) (ξ-μ)` — the metric-Gaussian log-density up to the
sample-independent `½logdet(I+F(μ))` constant. One `_posterior_metric` matvec.

EXACT for MGVI (the proposal is exactly `N(μ, (I+F(μ))⁻¹)`). For geoVI the draw bends each
sample through a per-sample nonlinear curve, so this is the underlying metric-Gaussian's
log-density (the curve Jacobian is dropped) — an approximate, but well-defined and
documented, diagnostic density.
"""
function logdensity_unnormalized(d::FisherGaussianDistribution, ξ)
    δ = ξ .- d.mean
    return -0.5 * real(dot(δ, _posterior_metric(d.likelihood, d.mean, δ)))
end

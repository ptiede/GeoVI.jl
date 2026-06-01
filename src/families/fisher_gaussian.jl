# ── Shared Fisher-Gaussian machinery ─────────────────────────────────────────
# MGVI and geoVI are the same family — a Gaussian with covariance `(I + Fisher)⁻¹`,
# sampled by a CG solve. geoVI only adds a nonlinear coordinate curve to the draw. They
# share every method except that one refinement step (`_refine_residual`, defined in
# `mgvi.jl` / `geovi.jl`), so the draw, the two-space noise bundle, and the
# natural-gradient metric all dispatch on this union.

const FisherGaussian = Union{MGVIFamily, GeoVIFamily}

# Two-space white noise: `metric` in the likelihood tangent space, `prior` in latent space.
init_noise(::FisherGaussian, adtype, lh::AbstractLikelihood, latent, n) = (;
    metric = (
        let t = _tangent_template(adtype, lh, latent)
            similar(t, (n, size(t)...))
        end
    ),
    prior = similar(latent, (n, size(latent)...)),
)
draw_noise(::FisherGaussian, lh::AbstractLikelihood, μ, rng::AbstractRNG) = (;
    metric = randn_like(rng, _metric_tangent_template(lh, μ)),
    prior = randn_like(rng, μ),
)

# Frozen draw: build the metric sample from the white noise, CG-solve for the linear
# residual, then refine (MGVI: none; geoVI: the curve).
function transform_block(fam::FisherGaussian, lh::AbstractLikelihood, μ, noise_i, mirrored::Bool)
    ms = _metric_sample_from_white(lh, μ, noise_i.metric, noise_i.prior)
    linear = draw_linear_residual(lh, μ, ms; _draw_linear_kwargs(fam.solver)...)
    return _refine_residual(fam, lh, μ, linear, ms, mirrored)
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

    result = zero(v)
    n = _sample_count(residuals)
    @trace track_numbers = false for i in 1:n
        result = result .+ _posterior_metric(
            lh,
            transport(family, θ, _sample_slice(residuals, i)),
            v,
        ) ./ n
    end
    return result
end

# ── The fitted distribution: shared by MGVI and geoVI ────────────────────────
# A draw is a CG solve against the posterior Fisher at the mean (geoVI adds the curve), so
# the distribution must carry the family (solver/curve) and the likelihood. Its `rand`
# reuses the same `draw_noise → transform_block → transport` pipeline as a fit step.
# There is no `logdensity` (the normalization is an intractable log-determinant).

struct FisherGaussianDistribution{F <: FisherGaussian, V, L} <: AbstractVariationalDistribution
    family::F
    mean::V
    likelihood::L
end

distribution(family::FisherGaussian, θ, likelihood) = FisherGaussianDistribution(family, θ, likelihood)

function Base.rand(rng::AbstractRNG, d::FisherGaussianDistribution)
    noise = draw_noise(d.family, d.likelihood, d.mean, rng)
    block = transform_block(d.family, d.likelihood, d.mean, noise, false)
    return transport(d.family, d.mean, _sample_slice(block, 1))
end

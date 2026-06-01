# ── MeanFieldGaussian ────────────────────────────────────────────────────────
# Mean-field ADVI — a pushforward family. It takes the default identity `transform_block`,
# the default single-latent-buffer `init_noise`, and overrides the differentiated suffix
# (`transport`), the log-density (`logdensity`), and the structured-θ helpers.

"""
    MeanFieldGaussian()

Mean-field automatic-differentiation VI (ADVI): the variational distribution is a
diagonal Gaussian `N(μ, diag(σ²))` in the latent (white) coordinates, with the
reparameterization `ξ = μ + σ ⊙ ε`, `ε ∼ N(0, I)`. The variational parameters are
the structured container `θ = (; mean = μ, logstd = log σ)` (so `σ` is a free
parameter, unlike MGVI/geoVI where the covariance is pinned by the Fisher metric).

Requires an `Optimisers.jl` rule (e.g. `Optimisers.Adam`) as the outer optimizer
and `n_samples > 0`; `NewtonCG` is not supported (no variational Fisher metric
yet). The objective is the ELBO, dispatched as `(MeanFieldGaussian, ReverseKL)`.
"""
struct MeanFieldGaussian <: AbstractVariationalFamily end

# θ = `(; mean = μ, logstd = log σ)`; the σ-scaling lives in `transport` (so `logstd` is
# differentiated through), while the frozen prefix stays the default identity (the stored
# residual is the raw white noise ε).
init_params(::MeanFieldGaussian, initial_latent) =
    (; mean = copy(initial_latent), logstd = zero(initial_latent))
transport(::MeanFieldGaussian, θ, ε) = θ.mean .+ exp.(θ.logstd) .* ε
draw_noise(::MeanFieldGaussian, lh::AbstractLikelihood, θ, rng::AbstractRNG) =
    randn_like(rng, θ.mean)

# `log q_θ(ξ)` for `q = N(μ, diag σ²)` with `ξ = μ + σ⊙ε`, i.e. `-½‖ε‖² - Σ logσ` (up to the
# global `-(d/2)·log2π` constant, dropped — it never affects an f-divergence's reduction).
# Only `-Σ logσ` carries a θ-gradient; `-½‖ε‖²` is per-sample but θ-independent (needed by
# nonlinear divergences like Rényi, gradient-free for reverse KL).
logdensity(::MeanFieldGaussian, θ, ε) = -sum(θ.logstd) - sum(abs2, ε) / 2

# ── The fitted distribution: a self-contained diagonal Gaussian ──────────────
# Needs no likelihood to sample, and (unlike the Fisher-Gaussian families) has a tractable
# `logdensity`, so it is a standalone `N(μ, diag σ²)` carrying only its parameters.

struct DiagonalGaussian{V} <: AbstractVariationalDistribution
    mean::V
    logstd::V
end

distribution(::MeanFieldGaussian, θ, likelihood) = DiagonalGaussian(θ.mean, θ.logstd)

Base.rand(rng::AbstractRNG, d::DiagonalGaussian) = d.mean .+ exp.(d.logstd) .* randn_like(rng, d.mean)

function logdensity(d::DiagonalGaussian, ξ)
    T = eltype(d.mean)
    ε = (ξ .- d.mean) .* exp.(-d.logstd)                  # standardize (the inverse map)
    return -sum(abs2, ε) / 2 - sum(d.logstd) - length(d.mean) * log(T(2π)) / 2
end

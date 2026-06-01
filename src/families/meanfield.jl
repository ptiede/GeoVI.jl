# ── MeanFieldGaussian ────────────────────────────────────────────────────────
# Mean-field ADVI — a pushforward family. It takes the default `draw_samples!` (the stored
# residual is raw white noise ε) and overrides only the differentiated suffix
# `transport_and_logjac` (which returns both the sample and its log-Jacobian) and `init_params`.

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

# θ = `(; mean = μ, logstd = log σ)`; the σ-scaling lives in `transport_and_logjac` (so
# `logstd` is differentiated through), while the frozen prefix stays the default
# `draw_samples!` (the stored residual is the raw white noise ε).
init_params(::MeanFieldGaussian, initial_latent) =
    (; mean = copy(initial_latent), logstd = zero(initial_latent))

# `ξ = μ + σ⊙ε`, with log-Jacobian `log|det diag(σ)| = Σ logσ` (the entropy term: the
# reverse-KL objective subtracts it, giving the familiar `-Σ logσ`).
transport_and_logjac(::MeanFieldGaussian, θ, ε) = (θ.mean .+ exp.(θ.logstd) .* ε, sum(θ.logstd))

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

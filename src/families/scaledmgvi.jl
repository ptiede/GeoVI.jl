# ── ScaledMGVI ───────────────────────────────────────────────────────────────
# A Fisher-Gaussian draw with a LEARNABLE diagonal rescaling on top. MGVI/geoVI pin the
# covariance to the Fisher metric `(I+F)⁻¹`; that metric can mis-scale individual latent
# directions (e.g. power-spectrum hyperparameters revert toward prior width). This family
# composes the inner Fisher-Gaussian metric draw `r ~ N(0,(I+F)⁻¹)` with a mean-field-style
# diagonal scaling `s = exp(logscale)`:
#
#   ξ = μ + s ⊙ r,   q = N(μ, diag(s) (I+F)⁻¹ diag(s)),   free params θ = [μ; logscale]
#
# θ is a FLAT array `[μ; logscale]`. It's a plain AbstractArray, so the NewtonCG optimizer +
# inner CG drive it UNCHANGED *and* with no wrapper quirks under Reactant (a ComponentArray
# carries its axis structure through the CG hot loop, where Reactant's broadcast/reduction/
# construction paths don't support it — norm, all, `similar`-with-axis, and the kw constructor
# each break). Structure is needed in only TWO structure-aware spots — the metric (block
# split) and transport (scale) — which split the flat array by index and `vcat` back (a traced
# concatenate). The `_mean`/`_logscale` view accessors keep that readable.
#
# `init`/`fit` take the LATENT mean `μ` (same as MGVI/mean-field) and `init_params` expands it
# to the full θ `[μ; 0]` — a single `init(rng, problem, μ)` call. `scaled_init` is the internal
# expander (also handy in tests for building a θ directly, incl. a seeded `scaled_init(μ, ℓ)`).

struct FullScale end

struct ScaledMGVI{F <: FisherGaussian, S} <: AbstractVariationalFamily
    inner::F
    sel::S
end
ScaledMGVI(inner::FisherGaussian = MGVIFamily(); sel = FullScale()) = ScaledMGVI(inner, sel)

# Split a flat θ (or tangent) into its mean / logscale views. FullScale ⇒ one scale per
# latent, so θ = [μ(D); logscale(D)] and D = length(θ) ÷ 2.
_mean(::FullScale, θ) = @view θ[1:(length(θ) ÷ 2)]
_logscale(::FullScale, θ) = @view θ[(length(θ) ÷ 2 + 1):end]
_mean(fam::ScaledMGVI, θ) = _mean(fam.sel, θ)
_logscale(fam::ScaledMGVI, θ) = _logscale(fam.sel, θ)

# Build the full starting θ from a latent mean (and optional logscale seed).
scaled_init(μ::AbstractArray) = vcat(μ, zero(μ))
scaled_init(μ::AbstractArray, logscale::AbstractArray) = vcat(μ, logscale)

# `init`/`fit` take the LATENT μ and expand it here to the full θ = [μ; 0] (logscale 0 ⇒ s = 1,
# so it starts as plain MGVI). A single `init(rng, problem, μ)` call — no separate `scaled_init`.
# (The no-ξ0 default path then just uses `default_latent(lh)` directly via `_default_xi0`'s
# fallback, which returns the latent unchanged.)
init_params(::ScaledMGVI, latent::AbstractArray) = scaled_init(latent)
# Seed BOTH components: pass a `(; mean, logscale)` NamedTuple. The flat θ = [mean; logscale]
# is built immediately, so the NamedTuple never enters the CG hot loop (where it would fight
# Reactant); it is purely a convenient init-input.
init_params(::ScaledMGVI, θ0::NamedTuple) = scaled_init(θ0.mean, θ0.logscale)

# The latent-shaped reference (the mean block) — for residual-buffer sizing and reset!.
_latent_ref(fam::ScaledMGVI, θ) = _mean(fam, θ)

# ξ = μ + s ⊙ r, with log-Jacobian Σ log s (the reverse-KL entropy term).
transport_and_logjac(fam::ScaledMGVI, θ, r) =
    (_mean(fam, θ) .+ exp.(_logscale(fam, θ)) .* r, sum(_logscale(fam, θ)))

# The stochastic phase: draw the inner metric residual r ~ N(0,(I+F)⁻¹) at μ (unscaled);
# the scaling is applied later in `transport_and_logjac`.
function draw_samples!(fam::ScaledMGVI, lh::AbstractLikelihood, θ, residuals, rng::AbstractRNG, mirrored::Bool)
    draw_samples!(fam.inner, lh, copy(_mean(fam, θ)), residuals, rng, mirrored)
    return residuals
end

supports_natural_gradient(::ScaledMGVI) = true

# Block-diagonal natural-gradient metric. The `mean` block reuses the inner Fisher-Gaussian
# metric `(I + mean_i F_i)` (so μ keeps EXACTLY the NewtonCG natural gradient it has under
# plain MGVI/geoVI); the `logscale` block is the identity. Assemble by `vcat` (a traced
# concatenate) so the result is a plain array, not a wrapped one.
function natural_gradient_metric(fam::ScaledMGVI, lh::AbstractLikelihood, θ, residuals, v::AbstractArray)
    mblock = natural_gradient_metric(fam.inner, lh, copy(_mean(fam, θ)), residuals, copy(_mean(fam, v)))
    return vcat(mblock, _logscale(fam, v))
end

# ── fitted distribution ──────────────────────────────────────────────────────
struct ScaledFisherGaussianDistribution{D, V, L, S} <: AbstractVariationalDistribution
    inner::D          # inner FisherGaussianDistribution at μ
    mean::V           # μ (so `q.mean` works for callers/_post_mean)
    logscale::L
    sel::S
end

function distribution(fam::ScaledMGVI, θ, lh)
    μ = copy(_mean(fam, θ))
    return ScaledFisherGaussianDistribution(distribution(fam.inner, μ, lh), μ, copy(_logscale(fam, θ)), fam.sel)
end

# A draw is μ + s ⊙ r, with r = (inner draw) − μ the inner metric residual.
function Base.rand(rng::AbstractRNG, d::ScaledFisherGaussianDistribution)
    full = rand(rng, d.inner)               # μ + r
    return d.mean .+ exp.(d.logscale) .* (full .- d.mean)
end

# ════════════════════════════════════════════════════════════════════════════
# Variational-family interface
#
# A family is a distribution `q_θ`, and VI minimizes the reverse KL
#   J(θ) = mean_i[ -log p(ξ_i) + log q_θ(ξ_i) ],   ξ_i = T_θ(η_i),  η_i ∼ N(0,I).
# `VIState.position` holds the variational parameters θ that the optimizer moves; the
# likelihood always consumes a latent point ξ. The reparameterization `T_θ` is sliced at
# the boundary between what is *frozen* at the current θ and what is *differentiated*:
#
#   draw_samples!        — the frozen prefix: fill the residual buffer with a Monte-Carlo
#                          set drawn at the current θ and held constant during the update.
#                          IID white noise for pushforward families; a CG solve (+ curve)
#                          for the Fisher-Gaussian families.
#   transport_and_logjac — the differentiated suffix, reconstructing ξ from θ + residual AND
#                          returning the reparameterization's log-Jacobian (its log q term).
#
# The VI loop is `draw_samples! → update!`: draw the samples once, then optimize θ against
# that fixed set (one `Optimisers` step, or `NewtonCG` to convergence), then resample on
# the next outer step — the same `draw → optimize-fixed-samples → resample` loop NIFTy uses.
#
# REQUIRED of a new family: `init_params` and (unless θ is itself the latent point)
# `transport_and_logjac`. Every other hook below has a default targeting the
# pushforward/common case (mean-field, normalizing flows). Concrete families live in the
# sibling files (`mgvi.jl`, `geovi.jl`, `fisher_gaussian.jl`, `meanfield.jl`).
# ════════════════════════════════════════════════════════════════════════════

"""
    AbstractVariationalFamily

Marker supertype for variational families. A family is the distribution `q_θ`; VI
minimizes the reverse KL `J(θ) = mean_i[-log p(ξ_i) + log q_θ(ξ_i)]` with
`ξ_i = T_θ(η_i)`, `η_i ∼ N(0,I)`. It carries the solvers its draw needs, but no
outer-optimization or Monte-Carlo-budget knobs — those live on the
[`MCEstimator`](@ref) estimator and the optimizer respectively.

To add a family, subtype `AbstractVariationalFamily` and implement
`GeoVI.init_params` and `GeoVI.transport_and_logjac` (the latter only if `θ` is not
itself the latent point). `transport_and_logjac(family, θ, r) -> (ξ, logjac)` returns both
the reconstructed latent point and the reparameterization's log-Jacobian `log|det J|` (the
variational entropy term). Optional hooks, each with a default targeting the pushforward
case (mean-field, normalizing flows): `GeoVI.draw_samples!` (the frozen draw; fills the
residual buffer with IID white noise by default), and — only to enable the `NewtonCG`
optimizer — `GeoVI.supports_natural_gradient` + `GeoVI.natural_gradient_metric`.
"""
abstract type AbstractVariationalFamily end

# ── Required (with pushforward-friendly defaults) ────────────────────────────

# Build θ from the user's initial latent point.
init_params(::AbstractVariationalFamily, initial_latent) = copy(initial_latent)

# Reconstruct a latent point ξ from θ + one stored residual, AND return the
# reparameterization's log-Jacobian `log|det J|` — the variational entropy term, so the
# reverse-KL objective is `mean_i[-log p(ξ_i) - logjac_i]`. Returns `(ξ, logjac)`, both
# DIFFERENTIATED through θ (so a structured θ stays differentiable, and a normalizing flow
# produces ξ and its `log|det J|` from one forward pass). Default: θ is itself the latent
# point, so `ξ = θ .+ residual` and `logjac = 0` — the Fisher-Gaussian fixed-metric
# approximation (the metric's log-determinant is intractable and held constant, so it drops).
transport_and_logjac(::AbstractVariationalFamily, θ, residual) = (θ .+ residual, zero(eltype(residual)))

# ── The frozen draw (`draw_samples!`) ────────────────────────────────────────
#
# `draw_samples!(family, lh, θ, residuals, rng, mirrored)` fills the preallocated
# `residuals` buffer (`(n_samples, latent…)`) with one Monte-Carlo set drawn at the current
# θ — the *frozen* part of the reparameterization, held constant during the `update!` that
# follows. The family owns its rng use and its loop; `mirrored` packs antithetic ±ε pairs
# (interleaved per base draw). The buffer's leading dimension is the sample count, so the
# number of base draws is `size(residuals, 1) ÷ (mirrored ? 2 : 1)`.

# Default (pushforward families — mean-field, normalizing flows): the frozen prefix is the
# identity, so the stored residual *is* white noise ε (the work happens in
# `transport_and_logjac`). The bulk `randn` is drawn OUTSIDE any `@trace` (the rng never
# enters a traced loop), so this compiles under Reactant unchanged.
function draw_samples!(
        ::AbstractVariationalFamily, lh::AbstractLikelihood, θ, residuals, rng::AbstractRNG,
        mirrored::Bool,
    )
    if !mirrored
        # One bulk draw fills the whole buffer (each row an independent ε).
        copyto!(residuals, randn_like(rng, residuals))
        return residuals
    end
    # Antithetic: draw the `n` base ε's in one shot (rng outside `@trace`), then write the
    # ±ε pairs — the `@trace for` slices/writes only (no rng inside the traced loop).
    n = size(residuals, 1) ÷ 2
    white = randn_like(rng, similar(residuals, (n, Base.tail(size(residuals))...)))
    @trace track_numbers = false for i in 1:n
        ε = _sample_slice(white, i)
        _write_sample_block!(residuals, i, _stack_residuals(ε, -ε))
    end
    return residuals
end

# ── Natural-gradient metric (an *optimizer* concern, not a family property) ──
#
# The metric is consumed *only* by `NewtonCG` (its inner CG solve and line-search
# curvature); a bare `Optimisers.jl` rule never asks for it. So it is not part of the
# core family interface: a family opts in to being usable with `NewtonCG` by setting
# `supports_natural_gradient` and implementing `natural_gradient_metric`.

supports_natural_gradient(::AbstractVariationalFamily) = false

# Default: undefined ⇒ this family cannot be used with `NewtonCG`.
function natural_gradient_metric(
        family::AbstractVariationalFamily, lh::AbstractLikelihood, θ, residuals, v::AbstractArray
    )
    throw(
        ArgumentError(
            "`$(nameof(typeof(family)))` defines no natural-gradient metric; `NewtonCG` " *
                "requires one. Use an `Optimisers.jl` rule, or implement " *
                "`GeoVI.natural_gradient_metric` (and `GeoVI.supports_natural_gradient`) for it.",
        ),
    )
end

# The metric comes in two layers, mirroring its geometry:
#
#   NaturalGradientField    — the metric *field* over θ-space: holds everything except the
#                             base point. Evaluating it at a base point x (once per Newton
#                             iteration) pins the field there, returning the operator on the
#                             tangent space at x. This is the curried `metricp` the
#                             optimizer contract consumes.
#   NaturalGradientOperator — the field pinned at one base point: the linear map
#                             `v ↦ (I + mean_i Fisher_i(base))·v` on the tangent space at
#                             `base`, applied once per inner-CG matvec.
#
# Pinning goes through the `_natural_gradient_operator` hook so a family can cache
# expensive per-base work (the Fisher-Gaussian families pin every sample point's
# forward-model linearization there); the default below just closes over the 5-arg
# `natural_gradient_metric`, re-deriving per application.

struct NaturalGradientOperator{F, L, B, R}
    family::F
    likelihood::L
    base::B
    residuals::R
end
function (op::NaturalGradientOperator)(v)
    return natural_gradient_metric(op.family, op.likelihood, op.base, op.residuals, v)
end

struct NaturalGradientField{F, L, R}
    family::F
    likelihood::L
    residuals::R
end
function (field::NaturalGradientField)(base)
    return _natural_gradient_operator(field.family, field.likelihood, base, field.residuals)
end

# The pinning hook, specialized by families that cache per-base work (see
# `fisher_gaussian.jl`); the fallback operator re-derives the metric per matvec.
function _natural_gradient_operator(
        family::AbstractVariationalFamily, lh::AbstractLikelihood, base, residuals
    )
    return NaturalGradientOperator(family, lh, base, residuals)
end


# ── The fitted variational distribution ─────────────────────────────────────
#
# Fitting produces parameters θ; bound to its family (and likelihood) those parameters
# *are* a distribution `q_θ`. `distribution` instantiates that first-class object — the
# exported output of `fit`. Each family returns its own concrete `AbstractVariational
# Distribution` subtype (in its file). `rand(rng, d[, n])` is the universal capability
# (always cheap to *call*; for the Fisher-Gaussian families a draw is a CG solve);
# `logdensity` is optional — defined only where the density is tractable.

abstract type AbstractVariationalDistribution end

"""
    distribution(family, θ, likelihood) -> AbstractVariationalDistribution

Instantiate the variational distribution `q_θ` for `family` at parameters `θ` (the
`likelihood` is carried for families whose draw needs it, e.g. MGVI/geoVI). Works at any
`θ`, not just a fitted one. Each family implements its own method returning its concrete
distribution type; see also `distribution(problem, state)`.
"""
function distribution(family::AbstractVariationalFamily, θ, likelihood)
    throw(
        ArgumentError(
            "`$(nameof(typeof(family)))` does not define `distribution`; implement " *
                "`GeoVI.distribution(::$(nameof(typeof(family))), θ, likelihood)`.",
        ),
    )
end

"""
    logdensity(d::AbstractVariationalDistribution, ξ) -> Real

The variational log-density `log q(ξ)` at a latent point `ξ` (mirrors `logdensity(lh, ξ)`
for the model). Optional: defined only for families with a tractable density (mean-field;
not MGVI/geoVI, whose normalization is an intractable log-determinant). `rand` is always
available even when this is not.
"""
function logdensity(d::AbstractVariationalDistribution, ξ)
    throw(
        ArgumentError(
            "`$(nameof(typeof(d)))` has no tractable `logdensity` (only `rand` is available).",
        ),
    )
end

# Sizing template for batched `rand`: same shape/eltype as one draw, obtained WITHOUT
# drawing (so `rand(rng, d, 0)` never advances the rng). Defaults to the distribution's
# `mean` field; a distribution without one overrides `_sample_template`.
_sample_template(d::AbstractVariationalDistribution) = d.mean

# Shared: `n` independent draws stacked along a leading axis (each row a latent point).
Base.rand(d::AbstractVariationalDistribution) = rand(Random.default_rng(), d)
Base.rand(d::AbstractVariationalDistribution, n::Integer) = rand(Random.default_rng(), d, n)
function Base.rand(rng::AbstractRNG, d::AbstractVariationalDistribution, n::Integer)
    n >= 0 || throw(ArgumentError("`n` must be non-negative"))
    template = _sample_template(d)
    out = similar(template, (n, size(template)...))
    trailing = ntuple(_ -> Colon(), ndims(template))
    @trace track_numbers = false for i in 1:n
        out[i, trailing...] = rand(rng, d)
    end
    return out
end

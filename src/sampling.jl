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
    MCEstimator(; n_samples=8, mirrored=true)

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

function MCEstimator(; n_samples = 8, mirrored = true)
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
# Metric-space white noise must have unit variance per REAL degree of freedom so that
# `leftsqrtmetric(lh, xi, ·)` produces a metric sample with covariance exactly `F = JᵀMJ`.
# `randn` on a COMPLEX tangent space (the visibility likelihood's `normalized_residual`) gives
# variance ½ per real component (E|w|² = 1), so the complex case is rescaled by √2 to recover
# the same convention the real-data path already satisfies. Without this the complex metric
# sample is `F/2`, the draw covariance is mis-scaled, and posterior draws are under-dispersed.
_unit_real_variance_white(w::AbstractArray{<:Complex}) = w .* sqrt(real(eltype(w))(2))
_unit_real_variance_white(w::AbstractArray) = w

function _metric_sample_from_white(lh::AbstractLikelihood, xi, metric_white, prior_white)
    likelihood_sample = leftsqrtmetric(lh, xi, _unit_real_variance_white(metric_white))
    return MetricSample(likelihood_sample .+ prior_white, prior_white)
end

# ── Draw-solve preconditioning ───────────────────────────────────────────────
#
# The draw solves `(I + F(xi)) r = b` by CG. When the likelihood is very
# informative (small noise) the data Fisher `F` has a few enormously stiff
# directions — `diag(F)` can span `1`…`1e8` — so the unpreconditioned system is
# wildly ill-conditioned and a capped CG leaves the stiff directions unresolved
# (the draw then rides the prior there, over-dispersing the corresponding
# parameters). A Jacobi preconditioner `M⁻¹ = diag(I+F)⁻¹` rescales every
# direction to O(1), collapsing the condition number so the existing iteration
# cap suffices. The prior block contributes exactly `1` to every diagonal entry,
# so `diag(I+F) ≥ 1` and `M⁻¹ ≤ I` (the preconditioner never amplifies).

"""
    JacobiPreconditioner(; n_probes = 24, floor = 1.0)

Request a diagonal (Jacobi) preconditioner for the Fisher-Gaussian draw solve.
The metric diagonal `diag(I+F(xi))` is estimated by Hutchinson's method with
`n_probes` Rademacher probes (one set of metric matvecs at the expansion point,
reused across every sample of the step), floored at `floor` (default `1.0`, the
exact prior lower bound `diag(I+F) ≥ 1`). Pass to a family, e.g.
`MGVIFamily(preconditioner = JacobiPreconditioner())`. `nothing` (the family
default) keeps the unpreconditioned draw.
"""
struct JacobiPreconditioner
    n_probes::Int
    floor::Float64
end
function JacobiPreconditioner(; n_probes::Integer = 24, floor::Real = 1.0)
    n_probes >= 1 || throw(ArgumentError("`n_probes` must be ≥ 1"))
    floor >= 0 || throw(ArgumentError("`floor` must be non-negative"))
    return JacobiPreconditioner(Int(n_probes), Float64(floor))
end

# The pinned diagonal operator: `v -> v .* dinv` (dinv = 1 ./ diag(I+F)).
struct _DiagPreconditioner{D}
    dinv::D
end
(p::_DiagPreconditioner)(v) = v .* p.dinv

_rademacher_like(rng, x) = sign.(randn_like(rng, x))

"""
    _posterior_metric_diag(lh, xi, rng, n_probes; floor = 1.0)

Hutchinson estimate of `diag(I + F(xi))`: `1 + mean_j[ z_j ⊙ (F z_j) ]` over
`n_probes` Rademacher probes `z_j`. `F z = fishermetric(lh, xi, z)` is the data
Fisher matvec (one pushforward+pullback through the forward model per probe).
Floored to stay `≥ floor` (MC noise can dip the estimate slightly negative).
"""
function _posterior_metric_diag(lh::AbstractLikelihood, xi, rng::AbstractRNG, n_probes::Integer; floor::Real = 1.0)
    probes = _rademacher_like(rng, similar(xi, (n_probes, size(xi)...)))
    acc = zero(xi)
    # `@trace for` → one MLIR while-loop body (matches the sample/diag loops),
    # not `n_probes` unrolled metric matvecs. In-place `.+=` mirrors
    # `natural_gradient_metric`'s accumulation idiom.
    @trace track_numbers = false for j in 1:n_probes
        z = _sample_slice(probes, j)
        acc .+= z .* fishermetric(lh, xi, z)
    end
    acc ./= n_probes
    # diag(I+F): add the prior's `1`, floor at the exact lower bound diag(I+F) ≥ 1
    # (MC noise can dip a raw entry slightly below).
    return max.(acc .+ one(eltype(acc)), oftype(one(eltype(acc)), floor))
end

# ── Low-rank deflation preconditioner ────────────────────────────────────────
#
# A diagonal preconditioner only helps a diagonally-dominant metric. When the
# data Fisher `F` is strongly COUPLED (a few stiff eigenmodes that mix many
# latent directions — e.g. a global amplitude/gauge mode spanning the whole
# image), its diagonal misrepresents the conditioning and Jacobi can make CG
# *worse*. The fix is to deflate `F`'s top eigenmodes directly. We estimate them
# with a randomized eigendecomposition (only `matmul` + `svd`, the dense ops
# available under Reactant), then apply
#
#   M⁻¹ x = x − Σ_j (λ_j/(1+λ_j)) v_j (v_jᵀx)
#
# which maps each captured stiff mode `v_j` of `(I+F)` to eigenvalue `1` and
# leaves the (well-conditioned) orthogonal complement untouched, collapsing the
# condition number CG sees. Storage is the `rank×D` basis `V` — `O(D)` memory
# (a handful of parameter-vectors), not the `O(D²)` dense factor.

"""
    DeflationPreconditioner(; rank = 20, oversample = 10)

Request a low-rank deflation preconditioner for the Fisher-Gaussian draw solve.
A randomized eigendecomposition (`rank + oversample` metric matvecs at the
expansion point, reused across every sample of the step) captures the top
`rank` eigenmodes of the data Fisher `F`, and the preconditioner deflates them
in `(I+F)`. Use this — not [`JacobiPreconditioner`](@ref) — when `F` is strongly
coupled (a diagonal preconditioner then fails or worsens CG). Memory is the
`(rank+oversample)×D` basis (a few parameter-vectors), scalable to millions of
latents. Pass to a family, e.g.
`MGVIFamily(preconditioner = DeflationPreconditioner(rank = 30))`.
"""
struct DeflationPreconditioner
    rank::Int
    oversample::Int
end
function DeflationPreconditioner(; rank::Integer = 20, oversample::Integer = 10)
    rank >= 1 || throw(ArgumentError("`rank` must be ≥ 1"))
    oversample >= 0 || throw(ArgumentError("`oversample` must be non-negative"))
    return DeflationPreconditioner(Int(rank), Int(oversample))
end

# The pinned low-rank operator. `V` is `ℓ×D` (each ROW an eigenvector of `F`,
# matching the `ℓ×D` block layout the sampler uses), `coef[j] = λ_j/(1+λ_j)`.
# `M⁻¹ x = x − Vᵀ (coef ⊙ (V x))`.
struct _LowRankPreconditioner{V, C}
    V::V
    coef::C
end
function (p::_LowRankPreconditioner)(x)
    proj = p.V * x                      # ℓ-vector: vⱼᵀx
    return x .- p.V' * (p.coef .* proj)
end

# ── Curve preconditioner ─────────────────────────────────────────────────────
# The geoVI curve's Gauss-Newton metric `(I + L_x R_ξ̄)(I + L_ξ̄ R_x)` equals `(I+F)²`
# at the expansion point, so its stiff modes have eigenvalue `(1+λ)²` — the SQUARE of
# `(I+F)`'s. A preconditioner built for the *linear draw* (`(I+F)`, coef `λ/(1+λ)`) only
# square-roots that conditioning (κ ~ 1e16 → 1e8) and leaves the curve's inner CG starved.
# To map a captured stiff mode of the curve metric to eigenvalue 1 we need
# `coef_curve = 1 - 1/(1+λ)²`. The linear deflation stores `coef = λ/(1+λ)`, so
# `1/(1+λ) = 1 - coef` and `coef_curve = 1 - (1-coef)²` — same eigenvectors `V`, reused.
# (Jacobi: the curve diagonal is `diag(I+F)²`, so `dinv_curve = dinv²`.)
_curve_preconditioner(::Nothing) = nothing
_curve_preconditioner(p::_DiagPreconditioner) = _DiagPreconditioner(p.dinv .^ 2)
function _curve_preconditioner(p::_LowRankPreconditioner)
    o = one(eltype(p.coef))
    return _LowRankPreconditioner(p.V, o .- (o .- p.coef) .^ 2)
end

# Apply `F` to every row of an `ℓ×D` block `Ω`, returning the `ℓ×D` block whose
# row j is `fishermetric(lh, xi, Ω_j)`. One `@trace for` over the ℓ rows (mirrors
# `_posterior_metric_diag` / `draw_samples!`), so it compiles to a single loop
# body rather than ℓ unrolled metric matvecs.
function _fisher_block(lh::AbstractLikelihood, xi, Ω)
    ℓ = size(Ω, 1)
    out = similar(Ω)
    @trace track_numbers = false for j in 1:ℓ
        fj = fishermetric(lh, xi, _sample_slice(Ω, j))
        _write_sample_block!(out, j, _single_sample_block(fj))
    end
    return out
end

# Symmetric ℓ×ℓ eigendecomposition via `svd` (Reactant has no `eigen`): for a
# symmetric PSD `B`, `svd(B) = U diag(λ) Uᵀ`, so the singular values ARE the
# eigenvalues and the left singular vectors the eigenvectors. `B` is symmetrized
# first to kill the small asymmetry from finite-precision matmuls.
function _sym_eig(B)
    Bs = (B .+ B') ./ 2
    F = svd(Bs)
    return F.U, F.S
end

# Build the per-step preconditioner once at the expansion point (or `nothing`).
_build_preconditioner(::Nothing, lh, xi, rng) = nothing
function _build_preconditioner(jp::JacobiPreconditioner, lh::AbstractLikelihood, xi, rng::AbstractRNG)
    d = _posterior_metric_diag(lh, xi, rng, jp.n_probes; floor = jp.floor)
    return _DiagPreconditioner(inv.(d))
end
function _build_preconditioner(dp::DeflationPreconditioner, lh::AbstractLikelihood, xi, rng::AbstractRNG)
    ℓ = dp.rank + dp.oversample
    # Randomized range finding for F's top eigenspace (blocks are ℓ×D — each row a
    # latent-shaped vector — so the linear algebra below is all `matmul` + small `svd`).
    Ω = randn_like(rng, similar(xi, (ℓ, size(xi)...)))
    Y = _fisher_block(lh, xi, Ω)                       # row j = F Ω_j
    # Orthonormalize the rows of Y. The Gram route `Q = (YYᵀ)^{-1/2} Y` loses
    # precision when Y has huge dynamic range (the stiff modes make `YYᵀ` span
    # ~`λ_max²`); an `svd` of `Yᵀ` (the only stable factorization Reactant offers)
    # gives a cleanly orthonormal basis. `svd(Yᵀ).U` is `D×ℓ` with orthonormal
    # columns ⇒ its transpose has orthonormal rows.
    Q = permutedims(svd(Y').U)                          # ℓ×D, orthonormal rows
    # Rayleigh–Ritz: B = Q F Qᵀ (ℓ×ℓ); its eigenpairs are F's top (λ_j, v_j).
    Z = _fisher_block(lh, xi, Q)                        # row j = F Q_j
    B = Q * Z'                                          # ℓ×ℓ symmetric PSD
    Ub, λ = _sym_eig(B)                                 # eigenpairs, λ descending
    # Keep only the top `rank` modes and DISCARD the `oversample` extras: the
    # oversamples sharpen the range-finding of the kept modes, but they are not
    # well-separated from the uncaptured bulk, so deflating them imperfectly would
    # inject a spurious small eigenvalue and re-inflate κ. (svd orders λ
    # descending, so `1:rank` is the stiff end.)
    keep = 1:dp.rank
    V = (Ub' * Q)[keep, :]                              # rank×D, top eigenvectors
    coef = λ[keep] ./ (one(eltype(λ)) .+ λ[keep])       # λ/(1+λ) ∈ [0,1)
    return _LowRankPreconditioner(V, coef)
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
        preconditioner = nothing,
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
        preconditioner = preconditioner,
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
        preconditioner = nothing,
    )
    cg = ConjugateGradient(rtol = cg_rtol, atol = cg_atol, maxiter = cg_maxiter, miniter = cg_miniter)
    residual, info = solve(cg, _PosteriorMetricOperator(lh, xi), metric_sample.metric; x0 = metric_sample.prior, preconditioner = preconditioner)

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
        preconditioner = nothing,
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
        preconditioner = preconditioner,
    )
end

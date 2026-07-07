# ── MGVIFamily ───────────────────────────────────────────────────────────────
# Metric Gaussian VI. The shared Fisher-Gaussian draw/noise/metric machinery lives in
# `fisher_gaussian.jl`; MGVI's only specialization is that it does *no* refinement.

"""
    MGVIFamily(; solver=ConjugateGradient(rtol=1e-5, miniter=6), strict=false)

Metric Gaussian VI: the variational distribution is the Gaussian whose
covariance is the inverse posterior Fisher metric. A draw solves
`(I + Fisher) ξ = sample` with `solver`. Defaults track NIFTy.re's sampling
solver: `rtol=1e-5`, `miniter=6`, an iteration cap of `min(200, 20·D)` (see
[`solve`](@ref)), and `strict = false` so a draw that fails to reach tolerance is
kept rather than thrown (convergence info is still reported on standalone draws).
Set `strict = true` to instead abort the fit on a non-converged linear solve.
"""
struct MGVIFamily{S} <: AbstractVariationalFamily
    solver::S
    strict::Bool
end
# NIFTy.re-matched sampling solver (see GeoVIFamily): `rtol=1e-5` + `miniter=6`; the draw
# passes no forcing threshold, so `rtol` IS its convergence criterion (rtol=nothing ⇒ no
# criterion ⇒ runs to the `min(200, 20·D)` cap). `strict=false` mirrors NIFTy's non-raising
# default.
MGVIFamily(; solver = ConjugateGradient(rtol = 1.0e-5, miniter = 6), strict = false) = MGVIFamily(solver, strict)

# The frozen draw refinement (see `draw_samples!` in `fisher_gaussian.jl`): MGVI keeps
# the linear CG residual as-is.
_refine_residual(::MGVIFamily, lh, μ, linear, ms, mirrored) =
    mirrored ?
    _stack_residuals(linear.residual, -linear.residual) :
    _single_sample_block(linear.residual)

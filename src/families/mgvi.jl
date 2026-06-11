# ── MGVIFamily ───────────────────────────────────────────────────────────────
# Metric Gaussian VI. The shared Fisher-Gaussian draw/noise/metric machinery lives in
# `fisher_gaussian.jl`; MGVI's only specialization is that it does *no* refinement.

"""
    MGVIFamily(; solver=ConjugateGradient(), strict=true)

Metric Gaussian VI: the variational distribution is the Gaussian whose
covariance is the inverse posterior Fisher metric. A draw solves
`(I + Fisher) ξ = sample` with `solver`. With `strict = true` (the default) a
linear solve that fails to converge eagerly throws and aborts the fit; set
`strict = false` to keep the unconverged draw and continue (convergence info is
still reported on standalone draws).
"""
struct MGVIFamily{S} <: AbstractVariationalFamily
    solver::S
    strict::Bool
end
MGVIFamily(; solver = ConjugateGradient(), strict = true) = MGVIFamily(solver, strict)

# The frozen draw refinement (see `draw_samples!` in `fisher_gaussian.jl`): MGVI keeps
# the linear CG residual as-is.
_refine_residual(::MGVIFamily, lh, μ, linear, ms, mirrored) =
    mirrored ?
    _stack_residuals(linear.residual, -linear.residual) :
    _single_sample_block(linear.residual)

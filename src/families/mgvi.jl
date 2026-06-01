# ── MGVIFamily ───────────────────────────────────────────────────────────────
# Metric Gaussian VI. The shared Fisher-Gaussian draw/noise/metric machinery lives in
# `fisher_gaussian.jl`; MGVI's only specialization is that it does *no* refinement.

"""
    MGVIFamily(; solver=ConjugateGradient())

Metric Gaussian VI: the variational distribution is the Gaussian whose
covariance is the inverse posterior Fisher metric. A draw solves
`(I + Fisher) ξ = sample` with `solver`.
"""
struct MGVIFamily{S} <: AbstractVariationalFamily
    solver::S
end
MGVIFamily(; solver = ConjugateGradient()) = MGVIFamily(solver)

# The frozen draw refinement (see `draw_samples!` in `fisher_gaussian.jl`): MGVI keeps
# the linear CG residual as-is.
_refine_residual(::MGVIFamily, lh, μ, linear, ms, mirrored) =
    mirrored ?
    _stack_residuals(linear.residual, -linear.residual) :
    _single_sample_block(linear.residual)

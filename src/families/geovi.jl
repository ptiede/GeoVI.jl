# ── GeoVIFamily ──────────────────────────────────────────────────────────────
# Geometric VI = MGVI + a nonlinear coordinate curve. It shares all of the
# Fisher-Gaussian machinery (`fisher_gaussian.jl`) and differs only in `_refine_residual`.

"""
    GeoVIFamily(; solver=ConjugateGradient(), curve=NewtonCG())

Geometric VI: a draw is an MGVI linear residual (`solver`) refined by a
nonlinear coordinate "curve" obtained by minimizing the geoVI residual
objective with `curve` (a [`NewtonCG`](@ref)). The curve is part of the *draw*,
not the outer fit.
"""
struct GeoVIFamily{S, C} <: AbstractVariationalFamily
    solver::S
    curve::C
end
GeoVIFamily(; solver = ConjugateGradient(), curve = NewtonCG()) = GeoVIFamily(solver, curve)

# The ONLY difference from MGVI: geoVI refines the linear residual with the nonlinear
# curve. (`throw_on_failure = false`: re-transforming at a converged mean can leave the
# curve unable to improve an already-optimal residual, which is not an error.)
function _refine_residual(fam::GeoVIFamily, lh, μ, linear, ms, mirrored)
    curve_options = _optimizer_kwargs(fam.curve)
    pos = update_nonlinear_residual(
        lh, μ, linear;
        optimizer = fam.curve, optimizer_options = curve_options, throw_on_failure = false,
    )
    mirrored || return _single_sample_block(pos.residual)
    neg = update_nonlinear_residual(
        lh, μ, -linear.residual;
        metric_sample = ms, metric_sample_sign = -1,
        optimizer = fam.curve, optimizer_options = curve_options, throw_on_failure = false,
    )
    return _stack_residuals(pos.residual, neg.residual)
end

# ── GeoVIFamily ──────────────────────────────────────────────────────────────
# Geometric VI = MGVI + a nonlinear coordinate curve. It shares all of the
# Fisher-Gaussian machinery (`fisher_gaussian.jl`) and differs only in `_refine_residual`.

"""
    GeoVIFamily(; solver=ConjugateGradient(), curve=NewtonCG(), strict=true)

Geometric VI: a draw is an MGVI linear residual (`solver`) refined by a
nonlinear coordinate "curve" obtained by minimizing the geoVI residual
objective with `curve` (a [`NewtonCG`](@ref)). The curve is part of the *draw*,
not the outer fit. With `strict = true` (the default) a linear solve that fails
to converge eagerly throws and aborts the fit; set `strict = false` to keep the
unconverged draw and continue. (The curve itself never throws — an
already-optimal residual that the curve cannot improve is not an error.)
"""
struct GeoVIFamily{S, C} <: AbstractVariationalFamily
    solver::S
    curve::C
    strict::Bool
end
GeoVIFamily(; solver = ConjugateGradient(), curve = NewtonCG(), strict = true) =
    GeoVIFamily(solver, curve, strict)

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

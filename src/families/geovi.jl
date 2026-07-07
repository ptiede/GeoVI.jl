# ── GeoVIFamily ──────────────────────────────────────────────────────────────
# Geometric VI = MGVI + a nonlinear coordinate curve. It shares all of the
# Fisher-Gaussian machinery (`fisher_gaussian.jl`) and differs only in `_refine_residual`.

"""
    GeoVIFamily(; solver=ConjugateGradient(rtol=1e-5, miniter=6), curve=NewtonCG(), strict=false)

Geometric VI: a draw is an MGVI linear residual (`solver`) refined by a
nonlinear coordinate "curve" obtained by minimizing the geoVI residual
objective with `curve` (a [`NewtonCG`](@ref)). The curve is part of the *draw*,
not the outer fit.

Defaults track NIFTy.re's sampling solver (`evi.draw_linear_residual` /
`conjugate_gradient._cg`): `rtol=1e-5`, `miniter=6`, an iteration cap of
`min(200, 20·D)` (see [`solve`](@ref)), and — via `strict = false` — a draw that
does NOT throw when CG fails to reach its tolerance. It keeps the (truncated)
residual and lets the curve refine it, exactly as NIFTy does with its default
`_raise_notconverged = False`. Set `strict = true` to instead abort the fit on a
non-converged linear solve. (The curve itself never throws — an already-optimal
residual it cannot improve is not an error.)
"""
struct GeoVIFamily{S, C, P} <: AbstractVariationalFamily
    solver::S
    curve::C
    strict::Bool
    preconditioner::P
end
# NIFTy.re-matched sampling solver: `rtol=1e-5` (a stochastic draw doesn't need the
# `ConjugateGradient` 1e-8 machine-precision default) and `miniter=6` (NIFTy's `_cg`
# floor). `strict=false` mirrors NIFTy's non-raising default — a draw that can't hit
# tolerance within the `min(200, 20·D)` cap is kept, not thrown. Override any of these
# for a tighter/stricter draw.
GeoVIFamily(; solver = ConjugateGradient(rtol = 1.0e-5, miniter = 6), curve = NewtonCG(), strict = false, preconditioner = nothing) =
    GeoVIFamily(solver, curve, strict, preconditioner)

# The ONLY difference from MGVI: geoVI refines the linear residual with the nonlinear
# curve. (`throw_on_failure = false`: re-transforming at a converged mean can leave the
# curve unable to improve an already-optimal residual, which is not an error.)
function _refine_residual(fam::GeoVIFamily, lh, μ, linear, ms, mirrored, precond = nothing)
    curve_options = _optimizer_kwargs(fam.curve)
    pos = update_nonlinear_residual(
        lh, μ, linear;
        optimizer = fam.curve, optimizer_options = curve_options, throw_on_failure = false,
        preconditioner = precond,
    )
    mirrored || return _single_sample_block(pos.residual)
    neg = update_nonlinear_residual(
        lh, μ, -linear.residual;
        metric_sample = ms, metric_sample_sign = -1,
        optimizer = fam.curve, optimizer_options = curve_options, throw_on_failure = false,
        preconditioner = precond,
    )
    return _stack_residuals(pos.residual, neg.residual)
end

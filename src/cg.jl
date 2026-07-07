struct ConjugateGradientInfo{C, I, R, B}
    converged::C
    iterations::I
    residual_norm::R
    breakdown::B
end

function _cg_info(; converged, iterations, residual_norm, breakdown = false)
    # `iterations` may be a plain `Int` (host call) or a `Reactant.TracedRNumber{Int}`
    # (inside a Reactant trace, after the `_maybe_traced` promotion in `_cg_run`).
    # Avoid eagerly casting to `Int` so both paths work; the struct is generic.
    return ConjugateGradientInfo(converged, iterations, residual_norm, breakdown)
end

function _check_denom(denom)
    valid_step = true
    breakdown = false
    @trace if denom <= 0
        valid_step = false
        breakdown = true
    end
    return valid_step, breakdown
end

function _check_conv(iteration, miniter, res_norm, tol)
    keep_going = true
    converged = false
    @trace if iteration >= miniter && res_norm <= tol
        keep_going = false
        converged = true
    end
    return keep_going, converged
end

# Preconditioned CG step. The loop-carried scalar is `rz = ⟨r, M⁻¹r⟩` (the
# preconditioned inner product); the convergence norm is the TRUE residual
# `‖r‖`. With `Minv = identity` (the default) `z = r`, `rz = ⟨r,r⟩`, and every
# line below reduces to the plain-CG recurrence — bit-for-bit. `z` is transient
# (recomputed from `r_new` each step), so it is never carried across iterations.
function _cg_step(denom, rz, x, r, p, Ap, Minv)
    x_new = x
    r_new = r
    p_new = p
    rz_new = rz
    residual_norm = sqrt(real(dot(r, r)))

    valid_step, breakdown = _check_denom(denom)
    @trace if valid_step
        α = rz / denom
        x_new = x .+ α .* p
        r_new = r .- α .* Ap
        z_new = Minv(r_new)
        rz_new = real(dot(r_new, z_new))
        residual_norm = sqrt(real(dot(r_new, r_new)))
        β = rz_new / rz
        p_new = z_new .+ β .* p
    end

    return x_new, r_new, p_new, rz_new, residual_norm, valid_step, breakdown
end

struct ConjugateGradient
    rtol::Union{Nothing, Float64}
    atol::Union{Nothing, Float64}
    maxiter::Union{Nothing, Int}
    miniter::Int
    absdelta::Union{Nothing, Float64}
end

"""
    ConjugateGradient(; rtol=1e-8, atol=0.0, maxiter=nothing, miniter=0, absdelta=nothing)

Matrix-free CG. Stops when the residual norm drops below `max(atol, rtol·‖b‖)`
(or an explicit per-call `threshold`). Either tolerance may be `nothing`,
meaning that criterion is absent (it contributes `0` to the `max`, so with both
absent the solve runs until `maxiter`/`absdelta` stop it). If `absdelta` is set
(or passed per call, as the Newton-CG outer loop does), CG also stops once the
decrease in its quadratic energy `φ(x) = ½xᵀAx − bᵀx` between iterations falls
below `absdelta` — a progress-based criterion that couples inner CG effort to
outer optimization gains.
"""
function ConjugateGradient(; rtol = 1.0e-8, atol = 0.0, maxiter = nothing, miniter = 0, absdelta = nothing)
    rtol === nothing || rtol >= 0 || throw(ArgumentError("`rtol` must be non-negative"))
    atol === nothing || atol >= 0 || throw(ArgumentError("`atol` must be non-negative"))
    miniter >= 0 || throw(ArgumentError("`miniter` must be non-negative"))
    maxiter !== nothing && maxiter < 0 && throw(ArgumentError("`maxiter` must be non-negative"))
    return ConjugateGradient(
        rtol === nothing ? nothing : Float64(rtol),
        atol === nothing ? nothing : Float64(atol),
        maxiter === nothing ? nothing : Int(maxiter),
        Int(miniter),
        absdelta === nothing ? nothing : Float64(absdelta),
    )
end

# A `nothing` tolerance contributes 0, i.e. the criterion is absent.
_tol_or_zero(tol) = tol === nothing ? 0.0 : float(tol)

function _cg_iterate(
        operator, b, rz, x, r, p, iteration, breakdown, miniter, ad_miniter, threshold, absdelta,
        energy, Minv,
    )
    Ap = operator(p)
    denom = real(dot(p, Ap))
    x, r, p, rz, residual_norm, valid_step, breakdown_step = _cg_step(denom, rz, x, r, p, Ap, Minv)
    iteration += ifelse(valid_step, 1, 0)
    breakdown = breakdown | breakdown_step
    keep_going, converged = _check_conv(iteration, miniter, residual_norm, threshold)
    # CG quadratic energy φ(x) = ½xᵀAx − bᵀx; with `r = b − Ax` this is
    # −½·Re⟨x, b + r⟩. Stop once its per-iteration decrease falls below
    # `absdelta`.
    new_energy = energy
    if absdelta !== nothing
        new_energy = -0.5 * real(dot(x, b .+ r))
        energy_diff = energy - new_energy
        @trace if (iteration >= ad_miniter) & (energy_diff < absdelta)
            keep_going = false
            converged = true
        end
    end
    keep_going = valid_step & keep_going
    return rz, x, r, p, iteration, keep_going, converged, breakdown, residual_norm, new_energy
end

function _cg_run(operator, b, maxiter::Int, miniter::Int, threshold, absdelta; x0 = nothing, preconditioner = nothing)
    # `Minv` applies the preconditioner `M⁻¹` (≈ A⁻¹) to a residual. The default
    # `identity` is plain CG: `z = r`, `rz = ⟨r,r⟩`, and the recurrence below is
    # unchanged. A Jacobi preconditioner passes `v -> v ./ diag(A)`.
    Minv = preconditioner === nothing ? identity : preconditioner
    x = x0 === nothing ? zero(b) : copy(x0)
    r = b .- operator(x)
    z = Minv(r)
    p = copy(z)

    rz = real(dot(r, z))
    residual_norm = sqrt(real(dot(r, r)))
    # Initial CG quadratic energy; only needed when the `absdelta` criterion is
    # active (the `dot` is skipped otherwise — see `_cg_iterate`). The
    # `=== nothing` guard is a compile-time type check, not a traced branch.
    energy = absdelta === nothing ? zero(rz) : -0.5 * real(dot(x, b .+ r))
    # Minimum iterations before the `absdelta` energy criterion may fire
    # (NIFTy.re uses `min(6, maxiter)`); honours a larger user `miniter`.
    ad_miniter = max(miniter, min(6, maxiter))

    # Promote loop-carried scalar state to TracedRNumber inside a Reactant
    # trace so the `@trace while` below can use `track_numbers=false`. With
    # `track_numbers=false`, Reactant won't try to derive traced versions of
    # any plain `Int`/`Bool` it encounters while walking the closure
    # environment (e.g. `LinRange.len::Int`, `Frequencies.n::Int`),
    # avoiding parametric-type-mismatch errors when the loop closure
    # reaches into a heavyweight forward model.
    iteration = _maybe_traced(0)
    keep_going = _maybe_traced(true)
    converged = _maybe_traced(false)
    breakdown = _maybe_traced(false)

    @trace if (residual_norm <= threshold) & (miniter == 0)
        keep_going = false
        converged = true
    end

    if maxiter == 0
        keep_going = false
    end

    @trace track_numbers = false while keep_going & (iteration < maxiter)
        rz, x, r, p, iteration, keep_going, converged, breakdown, residual_norm, energy =
            _cg_iterate(
            operator, b, rz, x, r, p, iteration, breakdown, miniter, ad_miniter, threshold,
            absdelta, energy, Minv,
        )
    end
    return x, _cg_info(
            converged = converged,
            iterations = iteration,
            residual_norm = residual_norm,
            breakdown = breakdown,
        )
end

# Identity outside a Reactant trace; inside, lifts a Julia scalar to a
# `TracedRNumber` so loop-carried state already has the trace-side
# representation before `@trace while`/`@trace for` is entered with
# `track_numbers=false`. Defined as a top-level (not closure-local)
# function so dispatch is stable inside the trace.
@inline _maybe_traced(x) = ReactantCore.within_compile() ?
    ReactantCore.promote_to_traced(x) : x

function solve(cg::ConjugateGradient, operator, b; x0 = nothing, threshold = nothing, absdelta = nothing, preconditioner = nothing)
    miniter = cg.miniter
    # Default iteration cap when `maxiter` is unset, matching NIFTy.re's `_cg`
    # (conjugate_gradient.py): `maxiter = max(min(200, 20·D), miniter)`. ONE cap for both
    # the plain residual draw and the inexact-Newton forcing solve — NIFTy drives both
    # through the same `_cg`. 200 is ample once the system is preconditioned/converging;
    # on an ill-conditioned draw it bounds the work (and, under `strict`, the failure)
    # instead of grinding toward ~2·D (~18k at D≈9k). `threshold` still selects the
    # forcing stop criterion below; it no longer scopes the iteration cap.
    maxiter = cg.maxiter === nothing ? max(miniter, min(200, 20 * length(b))) : cg.maxiter
    # An explicit `threshold` (e.g. an Eisenstat–Walker forcing term from the
    # Newton-CG outer loop) overrides the static `atol`/`rtol` criterion.
    thr = threshold === nothing ? max(_tol_or_zero(cg.atol), _tol_or_zero(cg.rtol) * norm(b)) : threshold
    # Per-call `absdelta` (e.g. a fraction of the last Newton energy gain)
    # overrides the static field; `nothing` (either source) disables the energy
    # criterion via the compile-time `=== nothing` guards in `_cg_run`.
    ad = absdelta === nothing ? cg.absdelta : absdelta
    return _cg_run(operator, b, maxiter, miniter, thr, ad; x0 = x0, preconditioner = preconditioner)
end

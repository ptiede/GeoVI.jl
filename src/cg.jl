struct ConjugateGradientInfo{C,I,R,B}
    converged::C
    iterations::I
    residual_norm::R
    breakdown::B
end

function _cg_info(; converged, iterations, residual_norm, breakdown=false)
    return ConjugateGradientInfo(converged, Int(iterations), residual_norm, breakdown)
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

function _cg_step(denom, rr, x, r, p, Ap)
    x_new = x
    r_new = r
    p_new = p
    rr_new = rr
    residual_norm = sqrt(rr)

    valid_step, breakdown = _check_denom(denom)
    @trace if valid_step
        α = rr / denom
        x_new = x .+ α .* p
        r_new = r .- α .* Ap
        rr_new = real(dot(r_new, r_new))
        residual_norm = sqrt(rr_new)
        β = rr_new / rr
        p_new = r_new .+ β .* p
    end

    return x_new, r_new, p_new, rr_new, residual_norm, valid_step, breakdown
end

struct ConjugateGradient
    rtol::Float64
    atol::Float64
    maxiter::Union{Nothing,Int}
    miniter::Int
end

function ConjugateGradient(; rtol=1e-8, atol=0.0, maxiter=nothing, miniter=0)
    rtol >= 0 || throw(ArgumentError("`rtol` must be non-negative"))
    atol >= 0 || throw(ArgumentError("`atol` must be non-negative"))
    miniter >= 0 || throw(ArgumentError("`miniter` must be non-negative"))
    maxiter !== nothing && maxiter < 0 && throw(ArgumentError("`maxiter` must be non-negative"))
    return ConjugateGradient(
        Float64(rtol), Float64(atol), maxiter === nothing ? nothing : Int(maxiter), Int(miniter)
    )
end

function _cg_iterate(operator, rr, x, r, p, iteration, breakdown, miniter, threshold)
    Ap = operator(p)
    denom = real(dot(p, Ap))
    x, r, p, rr, residual_norm, valid_step, breakdown_step = _cg_step(denom, rr, x, r, p, Ap)
    iteration += ifelse(valid_step, 1, 0)
    breakdown = breakdown | breakdown_step
    keep_going, converged = _check_conv(iteration, miniter, residual_norm, threshold)
    keep_going = valid_step & keep_going
    return rr, x, r, p, iteration, keep_going, converged, breakdown, residual_norm
end

function _cg_run(operator, b, maxiter::Int, miniter::Int, threshold; x0=nothing)
    x = x0 === nothing ? zero(b) : copy(x0)
    r = b .- operator(x)
    p = copy(r)

    rr = real(dot(r, r))
    residual_norm = sqrt(rr)

    iteration = 0
    keep_going = true
    converged = false
    breakdown = false

    @trace if (residual_norm <= threshold) & (miniter == 0)
        keep_going = false
        converged = true
    end

    if maxiter == 0
        keep_going = false
    end

    @trace while keep_going & (iteration < maxiter)
        rr, x, r, p, iteration, keep_going, converged, breakdown, residual_norm =
            _cg_iterate(operator, rr, x, r, p, iteration, breakdown, miniter, threshold)
    end
    return x, _cg_info(
        converged=converged,
        iterations=iteration,
        residual_norm=residual_norm,
        breakdown=breakdown,
    )
end

function solve(cg::ConjugateGradient, operator, b; x0=nothing)
    miniter = cg.miniter
    maxiter = cg.maxiter === nothing ? max(20, 2 * length(b)) : cg.maxiter
    threshold = max(float(cg.atol), float(cg.rtol) * norm(b))
    return _cg_run(operator, b, maxiter, miniter, threshold; x0=x0)
end

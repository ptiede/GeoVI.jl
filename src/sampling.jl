_tree_dofs(x::AbstractArray) = length(x)

_metric_tangent_template(lh::AbstractLikelihood, xi) = normalized_residual(lh, xi)
_posterior_metric(lh::AbstractLikelihood, xi, v) = fishermetric(lh, xi, v) .+ v

function _cg_info(; converged, iterations, residual_norm, breakdown=false)
    return (
        converged=converged,
        iterations=iterations,
        residual_norm=residual_norm,
        breakdown=breakdown,
    )
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

function _conjugate_gradient(
    operator,
    b;
    x0=nothing,
    rtol::Real=1e-8,
    atol::Real=0.0,
    maxiter::Union{Nothing,Integer}=nothing,
    miniter::Integer=0,
)
    rtol >= 0 || throw(ArgumentError("`rtol` must be non-negative"))
    atol >= 0 || throw(ArgumentError("`atol` must be non-negative"))
    miniter >= 0 || throw(ArgumentError("`miniter` must be non-negative"))

    if maxiter === nothing
        maxiter = max(20, 2 * _tree_dofs(b))
    else
        maxiter >= 0 || throw(ArgumentError("`maxiter` must be non-negative"))
    end

    x = x0 === nothing ? zero(b) : copy(x0)
    r = b .- operator(x)
    p = r

    rr = real(dot(r, r))
    residual_norm = sqrt(rr)
    threshold = max(float(atol), float(rtol) * norm(b))

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

        Ap = operator(p)
        denom = real(dot(p, Ap))
        x, r, p, rr, residual_norm, valid_step, breakdown_step =
            _cg_step(denom, rr, x, r, p, Ap)
        iteration += ifelse(valid_step, 1, 0)
        breakdown = breakdown | breakdown_step

        cg_keep_going, converged = _check_conv(iteration, miniter, residual_norm, threshold)
        keep_going = valid_step & cg_keep_going
    end
    return x, _cg_info(
        converged=converged,
        iterations=iteration,
        residual_norm=residual_norm,
        breakdown=breakdown,
    )
end

function _draw_metric_sample(lh::AbstractLikelihood, xi, rng::AbstractRNG)
    metric_white = randn_like(rng, _metric_tangent_template(lh, xi))
    likelihood_sample = leftsqrtmetric(lh, xi, metric_white)
    prior_sample = randn_like(rng, xi)
    return likelihood_sample .+ prior_sample, prior_sample
end

"""
    draw_linear_residual(lh, xi, rng; kwargs...)

Draw a linear MGVI residual around expansion point `xi`.

The draw is performed in white latent coordinates, so the covariance of the
returned residual is the inverse posterior Fisher metric

`I + fishermetric(lh, xi, ·)`.

If `from_inverse=false`, the function returns a draw with covariance equal to
the posterior Fisher metric itself instead.
"""
function draw_linear_residual(
    lh::AbstractLikelihood,
    xi,
    rng::AbstractRNG;
    from_inverse::Bool=true,
    metric_sample=nothing,
    cg_rtol::Real=1e-8,
    cg_atol::Real=0.0,
    cg_maxiter::Union{Nothing,Integer}=nothing,
    cg_miniter::Integer=0,
    x0=nothing,
    throw_on_failure::Bool=true,
)
    prior_sample = x0
    if metric_sample === nothing
        metric_sample, prior_sample = _draw_metric_sample(lh, xi, rng)
    end

    if !from_inverse
        return (
            metric_sample,
            _cg_info(converged=true, iterations=0, residual_norm=0.0, breakdown=false),
        )
    end

    operator = v -> _posterior_metric(lh, xi, v)
    residual, info = _conjugate_gradient(
        operator,
        metric_sample;
        x0=(prior_sample === nothing ? zero(metric_sample) : prior_sample),
        rtol=cg_rtol,
        atol=cg_atol,
        maxiter=cg_maxiter,
        miniter=cg_miniter,
    )

    if throw_on_failure && !info.converged
        throw(
            ErrorException(
                "conjugate gradient failed to converge after $(info.iterations) iterations",
            ),
        )
    end

    return residual, info
end

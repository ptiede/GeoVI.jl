_option(options, name::Symbol, default) =
    hasproperty(options, name) ? getproperty(options, name) : default

"""
    AbstractOptimizer

Marker supertype for built-in optimizer backends.

To add a new optimizer backend, subtype `AbstractOptimizer` and implement
`_optimize(optimizer, x0; fun_and_grad, hessp, kwargs...)`. If the backend
needs persistent state across VI iterations, also implement
`_optimizer_state(optimizer, x0, previous_result)`.

Any `Optimisers.AbstractRule` is also accepted directly without subtyping
`AbstractOptimizer`.
"""
abstract type AbstractOptimizer end
struct NewtonCG <: AbstractOptimizer end

struct OptimizationResult{OPT,S,X,C,K,ST,V,G,I,OE,HE,LS}
    optimizer::OPT
    optimizer_state::S
    x::X
    converged::C
    skipped::K
    status::ST
    value::V
    gradient::G
    iterations::I
    objective_evaluations::OE
    hessian_evaluations::HE
    line_search_steps::LS
end

function _optimization_result(
    optimizer;
    x,
    converged::Bool,
    status,
    value,
    gradient,
    iterations::Integer,
    objective_evaluations::Integer,
    hessian_evaluations::Integer=0,
    line_search_steps::Integer=0,
    optimizer_state=nothing,
    skipped::Bool=false,
)
    return OptimizationResult(
        optimizer,
        optimizer_state,
        x,
        converged,
        skipped,
        status,
        value,
        gradient,
        Int(iterations),
        Int(objective_evaluations),
        Int(hessian_evaluations),
        Int(line_search_steps),
    )
end

function _check_optimizer_conv(iteration, miniter, energy_diff, step_size, xtol, absdelta)
    keep_going = true
    converged = false

    if absdelta !== nothing
        @trace if (iteration > miniter) & (0 <= energy_diff) & (energy_diff < absdelta)
            keep_going = false
            converged = true
        end
    end

    @trace if keep_going & (iteration > miniter) & (step_size <= xtol)
        keep_going = false
        converged = true
    end

    return keep_going, converged
end

function _check_optimizer_step(new_x, step_size)
    valid = false
    @trace if all(isfinite, new_x) & isfinite(step_size)
        valid = true
    end
    return valid
end

function _check_objective_value(new_value)
    valid = false
    @trace if isfinite(new_value)
        valid = true
    end
    return valid
end

function _evaluate_optimizer_candidate(
    step_valid,
    value,
    grad,
    new_x,
    objective_evaluations,
    fun_and_grad,
)
    new_value = value
    new_grad = grad
    value_valid = false
    evaluations = objective_evaluations
    @trace if step_valid
        new_value, new_grad = fun_and_grad(new_x)
        evaluations += 1
        value_valid = _check_objective_value(new_value)
    end
    return new_value, new_grad, value_valid, evaluations
end

function _advance_optimizer_state(
    step_valid,
    value_valid,
    x,
    value,
    grad,
    new_x,
    new_value,
    new_grad,
    iteration,
    miniter,
    step_size,
    xtol,
    absdelta,
    status,
)
    x_out = x
    value_out = value
    grad_out = grad
    iteration_out = iteration
    keep_going = true
    converged = false
    status_out = status

    @trace if step_valid & value_valid
        iteration_out += 1
        energy_diff = value - new_value
        x_out, value_out, grad_out = new_x, new_value, new_grad
        keep_going, converged = _check_optimizer_conv(
            iteration_out,
            miniter,
            energy_diff,
            step_size,
            xtol,
            absdelta,
        )
        status_out = ifelse(converged, zero(status_out), status_out)
    else
        keep_going = false
        status_out = -1
    end

    return x_out, value_out, grad_out, iteration_out, keep_going, converged, status_out
end

_runtime_failure_enabled(x) = true

_optimizer_state(::Any, x0, previous_result) = nothing

function _optimizer_state(
    optimizer::Optimisers.AbstractRule,
    x0,
    previous_result,
)
    if previous_result !== nothing &&
        previous_result isa OptimizationResult &&
        previous_result.optimizer == optimizer &&
        previous_result.optimizer_state !== nothing
        return previous_result.optimizer_state
    end
    return Optimisers.setup(optimizer, x0)
end

function _prepare_optimizer_state(
    optimizer::Optimisers.AbstractRule,
    x0,
    optimizer_state,
)
    return isnothing(optimizer_state) ? Optimisers.setup(optimizer, x0) : optimizer_state
end

_optimizer_update(state, x, grad) = Optimisers.update(state, x, grad)

function _run_optimizer_rule(
    optimizer,
    state,
    x,
    value,
    grad;
    maxiter::Integer=20,
    miniter::Integer=0,
    xtol::Real=1e-5,
    absdelta=nothing,
    stepnorm=norm,
    objective_evaluations::Integer=1,
    evaluate_candidate,
    evaluation_state,
)
    iteration = 0
    keep_going = maxiter > 0
    converged = maxiter == 0
    status = converged ? 0 : maxiter

    @trace while keep_going & (iteration < maxiter)
        state, new_x = _optimizer_update(state, x, grad)
        delta = new_x .- x
        step_size = stepnorm(delta)
        step_valid = _check_optimizer_step(new_x, step_size)
        new_value, new_grad, value_valid, objective_evaluations = evaluate_candidate(
            step_valid,
            value,
            grad,
            new_x,
            objective_evaluations,
            evaluation_state,
        )
        x, value, grad, iteration, keep_going, converged, status = _advance_optimizer_state(
            step_valid,
            value_valid,
            x,
            value,
            grad,
            new_x,
            new_value,
            new_grad,
            iteration,
            miniter,
            step_size,
            xtol,
            absdelta,
            status,
        )
    end

    return _optimization_result(
        optimizer;
        x=x,
        converged=converged,
        status=status,
        value=value,
        gradient=grad,
        iterations=iteration,
        objective_evaluations=objective_evaluations,
        optimizer_state=state,
    )
end

_select_scalar(pred, a, b) = ifelse(pred, a, b)
function _select_array(pred, a, b)
    T = eltype(a)
    p = ifelse(pred, one(T), zero(T))
    return p .* a .+ (one(T) - p) .* b
end

function _line_search_step(
    ls_it, accepted, α, ls_steps, new_x, new_value, new_grad, direction, oe_inc,
    x, value, grad, fun_and_grad, valid_curve, curvature_direction,
)
    will_act = !accepted
    trial_x = x .- α .* direction
    trial_value, trial_grad = fun_and_grad(trial_x)

    inc = _select_scalar(will_act, 1, 0)
    out_oe_inc = oe_inc + inc
    out_ls_steps = ls_steps + inc

    is_good = isfinite(trial_value) & (trial_value <= value)
    will_accept = will_act & is_good
    out_accepted = accepted | will_accept

    out_new_x = _select_array(will_accept, trial_x, new_x)
    out_new_value = _select_scalar(will_accept, trial_value, new_value)
    out_new_grad = _select_array(will_accept, trial_grad, new_grad)

    bad = will_act & !is_good
    α_after_halve = _select_scalar(bad, α / 2, α)

    switch = bad & (ls_it == 5) & valid_curve
    out_α = _select_scalar(switch, one(α), α_after_halve)
    out_direction = _select_array(switch, curvature_direction, direction)

    return out_accepted, out_α, out_ls_steps, out_new_x, out_new_value, out_new_grad,
        out_direction, out_oe_inc
end

function _line_search(direction0, x, value, grad, fun_and_grad, hessp)
    α = one(eltype(x))
    accepted = false
    ls_steps = 0
    new_x = copy(x)
    new_value = value + zero(value)
    new_grad = copy(grad)
    direction = copy(direction0)
    oe_inc = 0

    Hg = hessp(x, grad)
    curvature = real(dot(grad, Hg))
    grad_norm_sq = real(dot(grad, grad))
    valid_curve = curvature > 0
    safe_curvature = _select_scalar(valid_curve, curvature, one(curvature))
    curvature_direction = (grad_norm_sq / safe_curvature) .* grad
    he_inc = 1

    if within_compile()
        α = promote_to_traced(α)
        accepted = promote_to_traced(accepted)
        ls_steps = promote_to_traced(ls_steps)
        oe_inc = promote_to_traced(oe_inc)
    end

    @trace for ls_it in 0:8
        (accepted, α, ls_steps, new_x, new_value, new_grad, direction, oe_inc) =
            _line_search_step(
                ls_it, accepted, α, ls_steps, new_x, new_value, new_grad, direction, oe_inc,
                x, value, grad, fun_and_grad, valid_curve, curvature_direction,
            )
    end

    return new_x, new_value, new_grad, accepted, α, direction, ls_steps, he_inc, oe_inc
end

function _newton_cg_iter(
    active, x, value, grad, status, iterations, converged,
    objective_evaluations, hessian_evaluations, line_search_steps,
    cg, hessp, fun_and_grad, stepnorm, miniter, xtol, absdelta, iteration,
)
    step, cg_info = solve(cg, v -> hessp(x, v), grad)
    cg_ok = cg_info.converged
    cg_iters = cg_info.iterations

    new_x_ls, new_value_ls, new_grad_ls, accepted, α, direction_used,
        ls_steps, he_inc, oe_inc =
        _line_search(step, x, value, grad, fun_and_grad, hessp)

    energy_diff = value - new_value_ls
    step_size = α * stepnorm(direction_used)
    done_by_energy = absdelta === nothing ?
        false :
        ((0 <= energy_diff) & (energy_diff < absdelta))
    convergence_hit = (iteration > miniter) & (done_by_energy | (step_size <= xtol))

    progressed = active & cg_ok & accepted
    deactivate_now = active & ((!cg_ok) | (cg_ok & !accepted) | (progressed & convergence_hit))

    new_active = active & !deactivate_now
    new_x = _select_array(progressed, new_x_ls, x)
    new_value = _select_scalar(progressed, new_value_ls, value)
    new_grad = _select_array(progressed, new_grad_ls, grad)

    cg_failed_now = active & !cg_ok
    ls_failed_now = active & cg_ok & !accepted
    converged_now = progressed & convergence_hit

    new_iterations = _select_scalar(
        cg_failed_now | ls_failed_now, iteration - 1,
        _select_scalar(progressed, iteration, iterations),
    )

    new_status = _select_scalar(
        cg_failed_now, -2,
        _select_scalar(
            ls_failed_now, -1,
            _select_scalar(converged_now, 0, status),
        ),
    )

    new_converged = converged | converged_now

    active_int = _select_scalar(active, 1, 0)
    cg_ok_int = _select_scalar(active & cg_ok, 1, 0)

    new_oe = objective_evaluations + active_int * (cg_ok_int * oe_inc)
    new_he = hessian_evaluations + active_int * (cg_iters + cg_ok_int * he_inc)
    new_lss = line_search_steps + cg_ok_int * ls_steps

    return new_active, new_x, new_value, new_grad, new_status, new_iterations, new_converged,
        new_oe, new_he, new_lss
end

function _optimize(
    optimizer::NewtonCG,
    x0::AbstractArray;
    fun_and_grad,
    hessp,
    maxiter::Integer=20,
    miniter::Integer=0,
    xtol::Real=1e-5,
    absdelta=nothing,
    cg_rtol::Real=1e-8,
    cg_atol::Real=0.0,
    cg_maxiter::Union{Nothing,Integer}=nothing,
    cg_miniter::Integer=0,
    stepnorm=norm,
    optimizer_state=nothing,
)
    maxiter >= 0 || throw(ArgumentError("`maxiter` must be non-negative"))
    miniter >= 0 || throw(ArgumentError("`miniter` must be non-negative"))

    x = x0
    value, grad = fun_and_grad(x)
    objective_evaluations = 1
    hessian_evaluations = 0
    line_search_steps = 0
    if maxiter == 0
        return _optimization_result(
            optimizer;
            x=x,
            converged=true,
            status=0,
            value=value,
            gradient=grad,
            iterations=0,
            objective_evaluations=objective_evaluations,
            hessian_evaluations=hessian_evaluations,
            line_search_steps=line_search_steps,
        )
    end

    cg = ConjugateGradient(rtol=cg_rtol, atol=cg_atol, maxiter=cg_maxiter, miniter=cg_miniter)
    converged = false
    status = maxiter
    iterations = 0
    active = true

    @trace for iteration in 1:maxiter
        (active, x, value, grad, status, iterations, converged,
         objective_evaluations, hessian_evaluations, line_search_steps) =
            _newton_cg_iter(
                active, x, value, grad, status, iterations, converged,
                objective_evaluations, hessian_evaluations, line_search_steps,
                cg, hessp, fun_and_grad, stepnorm, miniter, xtol, absdelta, iteration,
            )
    end

    return _optimization_result(
        optimizer;
        x=x,
        converged=converged,
        status=status,
        value=value,
        gradient=grad,
        iterations=iterations,
        objective_evaluations=objective_evaluations,
        hessian_evaluations=hessian_evaluations,
        line_search_steps=line_search_steps,
    )
end

function _optimize(
    optimizer::Optimisers.AbstractRule,
    x0::AbstractArray;
    fun_and_grad,
    hessp=nothing,
    maxiter::Integer=20,
    miniter::Integer=0,
    xtol::Real=1e-5,
    absdelta=nothing,
    cg_rtol::Real=1e-8,
    cg_atol::Real=0.0,
    cg_maxiter::Union{Nothing,Integer}=nothing,
    cg_miniter::Integer=0,
    stepnorm=norm,
    optimizer_state=nothing,
)
    maxiter >= 0 || throw(ArgumentError("`maxiter` must be non-negative"))
    miniter >= 0 || throw(ArgumentError("`miniter` must be non-negative"))

    state = _prepare_optimizer_state(optimizer, x0, optimizer_state)
    x = x0
    value, grad = fun_and_grad(x)
    return _run_optimizer_rule(
        optimizer,
        state,
        x,
        value,
        grad;
        maxiter=maxiter,
        miniter=miniter,
        xtol=xtol,
        absdelta=absdelta,
        stepnorm=stepnorm,
        objective_evaluations=1,
        evaluate_candidate=_evaluate_optimizer_candidate,
        evaluation_state=fun_and_grad,
    )
end

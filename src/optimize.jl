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
        if active
            step, cg_info = solve(cg, v -> hessp(x, v), grad)
            hessian_evaluations += cg_info.iterations

            if !cg_info.converged
                status = -2
                iterations = iteration - 1
                active = false
            else
                direction = step
                α = one(eltype(x))
                accepted = false
                ls_steps = 0
                new_x, new_value, new_grad = x, value, grad

                @trace for ls_it in 0:8
                    if !accepted
                        trial_x = x .- α .* direction
                        trial_value, trial_grad = fun_and_grad(trial_x)
                        objective_evaluations += 1
                        ls_steps += 1

                        if isfinite(trial_value) && trial_value <= value
                            new_x, new_value, new_grad = trial_x, trial_value, trial_grad
                            accepted = true
                        else
                            α /= 2
                            if ls_it == 5
                                Hg = hessp(x, grad)
                                hessian_evaluations += 1
                                curvature = real(dot(grad, Hg))
                                if curvature > 0
                                    α = one(α)
                                    direction = (real(dot(grad, grad)) / curvature) .* grad
                                end
                            end
                        end
                    end
                end
                line_search_steps += ls_steps

                if !accepted
                    status = -1
                    iterations = iteration - 1
                    active = false
                else
                    energy_diff = value - new_value
                    step_size = α * stepnorm(direction)
                    x, value, grad = new_x, new_value, new_grad
                    iterations = iteration

                    if iteration > miniter
                        done_by_energy =
                            absdelta === nothing ? false : (0 <= energy_diff < absdelta)
                        if done_by_energy || step_size <= xtol
                            converged = true
                            status = 0
                            active = false
                        end
                    end
                end
            end
        end
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

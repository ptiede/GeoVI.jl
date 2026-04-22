_option(options, name::Symbol, default) = hasproperty(options, name) ? getproperty(options, name) : default

function _nonlinear_residual_value_and_gradient(
    lh::AbstractLikelihood,
    expansion_point::AbstractArray,
    transformation_at_point::AbstractArray,
    metric_sample::AbstractArray,
    x::AbstractArray,
)
    t = transformation(lh, x) .- transformation_at_point
    g = x .- expansion_point .+ leftsqrtmetric(lh, expansion_point, t)
    r = metric_sample .- g
    value = 0.5 * real(dot(r, r))

    r̄ = conj.(r)
    grad = -(r̄ .+ leftsqrtmetric(lh, x, rightsqrtmetric(lh, expansion_point, r̄)))
    return value, grad
end

function _nonlinear_residual_metric(
    lh::AbstractLikelihood,
    expansion_point::AbstractArray,
    x::AbstractArray,
    v::AbstractArray,
)
    tm = leftsqrtmetric(lh, expansion_point, rightsqrtmetric(lh, x, v)) .+ v
    return leftsqrtmetric(lh, x, rightsqrtmetric(lh, expansion_point, tm)) .+ tm
end

function _nonlinear_residual_stepnorm(
    lh::AbstractLikelihood,
    expansion_point::AbstractArray,
    step::AbstractArray,
)
    pushed = rightsqrtmetric(lh, expansion_point, step)
    return sqrt(real(dot(step, step)) + real(dot(pushed, pushed)))
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
)
    return (
        optimizer=optimizer,
        optimizer_state=optimizer_state,
        x=x,
        converged=converged,
        status=status,
        value=value,
        gradient=gradient,
        iterations=Int(iterations),
        objective_evaluations=Int(objective_evaluations),
        hessian_evaluations=Int(hessian_evaluations),
        line_search_steps=Int(line_search_steps),
    )
end

_optimizer_state(::Any, x0, previous_result) = nothing

function _optimizer_state(
    optimizer::Optimisers.AbstractRule,
    x0,
    previous_result,
)
    if previous_result !== nothing &&
        hasproperty(previous_result, :optimizer) &&
        hasproperty(previous_result, :optimizer_state) &&
        previous_result.optimizer == optimizer &&
        previous_result.optimizer_state !== nothing
        return previous_result.optimizer_state
    end
    return Optimisers.setup(optimizer, x0)
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

    for iteration in 1:maxiter
        step, cg_info = _conjugate_gradient(
            v -> hessp(x, v),
            grad;
            rtol=cg_rtol,
            atol=cg_atol,
            maxiter=cg_maxiter,
            miniter=cg_miniter,
        )
        hessian_evaluations += cg_info.iterations

        if !cg_info.converged
            return _optimization_result(
                optimizer;
                x=x,
                converged=false,
                status=-2,
                value=value,
                gradient=grad,
                iterations=iteration - 1,
                objective_evaluations=objective_evaluations,
                hessian_evaluations=hessian_evaluations,
                line_search_steps=line_search_steps,
            )
        end

        direction = step
        α = 1.0
        accepted = false
        ls_steps = 0
        new_x, new_value, new_grad = x, value, grad

        for ls_it in 0:8
            trial_x = x .- α .* direction
            trial_value, trial_grad = fun_and_grad(trial_x)
            objective_evaluations += 1
            ls_steps += 1

            if isfinite(trial_value) && trial_value <= value
                new_x, new_value, new_grad = trial_x, trial_value, trial_grad
                accepted = true
                break
            end

            α /= 2
            if ls_it == 5
                Hg = hessp(x, grad)
                hessian_evaluations += 1
                curvature = real(dot(grad, Hg))
                if curvature > 0
                    α = 1.0
                    direction = (real(dot(grad, grad)) / curvature) .* grad
                end
            end
        end
        line_search_steps += ls_steps

        if !accepted
            return _optimization_result(
                optimizer;
                x=x,
                converged=false,
                status=-1,
                value=value,
                gradient=grad,
                iterations=iteration - 1,
                objective_evaluations=objective_evaluations,
                hessian_evaluations=hessian_evaluations,
                line_search_steps=line_search_steps,
            )
        end

        energy_diff = value - new_value
        step_size = α * stepnorm(direction)
        x, value, grad = new_x, new_value, new_grad

        if iteration > miniter
            if absdelta !== nothing && 0 <= energy_diff < absdelta
                return _optimization_result(
                    optimizer;
                    x=x,
                    converged=true,
                    status=0,
                    value=value,
                    gradient=grad,
                    iterations=iteration,
                    objective_evaluations=objective_evaluations,
                    hessian_evaluations=hessian_evaluations,
                    line_search_steps=line_search_steps,
                )
            end
            if step_size <= xtol
                return _optimization_result(
                    optimizer;
                    x=x,
                    converged=true,
                    status=0,
                    value=value,
                    gradient=grad,
                    iterations=iteration,
                    objective_evaluations=objective_evaluations,
                    hessian_evaluations=hessian_evaluations,
                    line_search_steps=line_search_steps,
                )
            end
        end
    end

    return _optimization_result(
        optimizer;
        x=x,
        converged=false,
        status=maxiter,
        value=value,
        gradient=grad,
        iterations=maxiter,
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

    state = isnothing(optimizer_state) ? Optimisers.setup(optimizer, x0) : optimizer_state
    x = x0
    value, grad = fun_and_grad(x)
    objective_evaluations = 1

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
            optimizer_state=state,
        )
    end

    for iteration in 1:maxiter
        state, new_x = Optimisers.update(state, x, grad)
        step = new_x .- x
        step_size = stepnorm(step)

        if !(all(isfinite, new_x) && isfinite(step_size))
            return _optimization_result(
                optimizer;
                x=x,
                converged=false,
                status=-1,
                value=value,
                gradient=grad,
                iterations=iteration - 1,
                objective_evaluations=objective_evaluations,
                optimizer_state=state,
            )
        end

        new_value, new_grad = fun_and_grad(new_x)
        objective_evaluations += 1

        if !isfinite(new_value)
            return _optimization_result(
                optimizer;
                x=x,
                converged=false,
                status=-1,
                value=value,
                gradient=grad,
                iterations=iteration - 1,
                objective_evaluations=objective_evaluations,
                optimizer_state=state,
            )
        end

        energy_diff = value - new_value
        x, value, grad = new_x, new_value, new_grad

        if iteration > miniter
            if absdelta !== nothing && 0 <= energy_diff < absdelta
                return _optimization_result(
                    optimizer;
                    x=x,
                    converged=true,
                    status=0,
                    value=value,
                    gradient=grad,
                    iterations=iteration,
                    objective_evaluations=objective_evaluations,
                    optimizer_state=state,
                )
            end
            if step_size <= xtol
                return _optimization_result(
                    optimizer;
                    x=x,
                    converged=true,
                    status=0,
                    value=value,
                    gradient=grad,
                    iterations=iteration,
                    objective_evaluations=objective_evaluations,
                    optimizer_state=state,
                )
            end
        end
    end

    return _optimization_result(
        optimizer;
        x=x,
        converged=false,
        status=maxiter,
        value=value,
        gradient=grad,
        iterations=maxiter,
        objective_evaluations=objective_evaluations,
        optimizer_state=state,
    )
end

function update_nonlinear_residual(
    lh::AbstractLikelihood,
    expansion_point::AbstractArray,
    residual_sample::AbstractArray;
    rng::Union{Nothing,AbstractRNG}=nothing,
    metric_sample=nothing,
    metric_sample_sign::Number=1,
    optimizer=NewtonCG(),
    optimizer_options=(;),
    throw_on_failure::Bool=true,
)
    if metric_sample === nothing
        rng === nothing &&
            throw(ArgumentError("specify either `rng` or `metric_sample`"))
        metric_sample, _ = _draw_metric_sample(lh, expansion_point, rng)
    end

    metric_sample = metric_sample_sign .* metric_sample
    sample0 = expansion_point .+ residual_sample

    maxiter = _option(optimizer_options, :maxiter, 20)
    if maxiter == 0
        return residual_sample,
        (
            x=sample0,
            converged=true,
            status=0,
            value=NaN,
            gradient=zero(sample0),
            iterations=0,
            objective_evaluations=0,
            hessian_evaluations=0,
            line_search_steps=0,
            skipped=true,
        )
    end

    transformation_at_point = transformation(lh, expansion_point)
    result = _optimize(
        optimizer,
        sample0;
        fun_and_grad=x -> _nonlinear_residual_value_and_gradient(
            lh,
            expansion_point,
            transformation_at_point,
            metric_sample,
            x,
        ),
        hessp=(x, v) -> _nonlinear_residual_metric(lh, expansion_point, x, v),
        maxiter=maxiter,
        miniter=_option(optimizer_options, :miniter, 0),
        xtol=_option(optimizer_options, :xtol, 1e-5),
        absdelta=_option(optimizer_options, :absdelta, nothing),
        cg_rtol=_option(optimizer_options, :cg_rtol, 1e-8),
        cg_atol=_option(optimizer_options, :cg_atol, 0.0),
        cg_maxiter=_option(optimizer_options, :cg_maxiter, nothing),
        cg_miniter=_option(optimizer_options, :cg_miniter, 0),
        stepnorm=step -> _nonlinear_residual_stepnorm(lh, expansion_point, step),
    )

    if throw_on_failure && !result.converged
        throw(
            ErrorException(
                "nonlinear residual update failed with status $(result.status) after $(result.iterations) iterations",
            ),
        )
    end

    return result.x .- expansion_point, result
end

function _stack_residuals(a::AbstractArray, b::AbstractArray)
    return cat(reshape(a, (1, size(a)...)), reshape(b, (1, size(b)...)); dims=1)
end

function draw_residual(
    lh::AbstractLikelihood,
    expansion_point::AbstractArray,
    rng::AbstractRNG;
    draw_linear_kwargs=(;),
    optimizer=NewtonCG(),
    optimizer_options=(;),
    throw_on_failure::Bool=true,
)
    metric_sample, prior_sample = _draw_metric_sample(lh, expansion_point, rng)
    linear_residual, linear_info = draw_linear_residual(
        lh,
        expansion_point,
        rng;
        draw_linear_kwargs...,
        metric_sample=metric_sample,
        x0=prior_sample,
        throw_on_failure=throw_on_failure,
    )

    positive_residual, positive_info = update_nonlinear_residual(
        lh,
        expansion_point,
        linear_residual;
        metric_sample=metric_sample,
        metric_sample_sign=1,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
        throw_on_failure=throw_on_failure,
    )
    negative_residual, negative_info = update_nonlinear_residual(
        lh,
        expansion_point,
        -linear_residual;
        metric_sample=metric_sample,
        metric_sample_sign=-1,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
        throw_on_failure=throw_on_failure,
    )

    return _stack_residuals(positive_residual, negative_residual),
    (linear=linear_info, positive=positive_info, negative=negative_info)
end

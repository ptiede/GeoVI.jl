struct NonlinearResidualUpdate{R,O}
    residual::R
    result::O
end

struct MirroredResidualDraw{R,L,N}
    residuals::R
    linear::L
    positive::N
    negative::N
end

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

function _metric_sample_array(metric_sample)
    return metric_sample isa MetricSample ? metric_sample.metric : metric_sample
end

function _stack_residuals(a::AbstractArray, b::AbstractArray)
    return cat(reshape(a, (1, size(a)...)), reshape(b, (1, size(b)...)); dims=1)
end

function update_nonlinear_residual(
    lh::AbstractLikelihood,
    expansion_point::AbstractArray,
    linear_draw::LinearResidualDraw;
    metric_sample_sign::Number=1,
    optimizer=NewtonCG(),
    optimizer_options=(;),
    throw_on_failure::Bool=true,
)
    return update_nonlinear_residual(
        lh,
        expansion_point,
        linear_draw.residual;
        metric_sample=linear_draw.metric_sample,
        metric_sample_sign=metric_sample_sign,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
        throw_on_failure=throw_on_failure,
    )
end

function update_nonlinear_residual(
    lh::AbstractLikelihood,
    expansion_point::AbstractArray,
    residual_sample::AbstractArray;
    metric_sample=nothing,
    metric_sample_sign::Number=1,
    optimizer=NewtonCG(),
    optimizer_options=(;),
    throw_on_failure::Bool=true,
)
    metric_sample === nothing && throw(
        ArgumentError(
            "pass either a `LinearResidualDraw` or the explicit `metric_sample` used to construct the linear residual",
        ),
    )

    metric_sample_array = metric_sample_sign .* _metric_sample_array(metric_sample)
    sample0 = expansion_point .+ residual_sample

    maxiter = _option(optimizer_options, :maxiter, 20)
    if maxiter == 0
        result = _optimization_result(
            optimizer;
            x=sample0,
            converged=true,
            skipped=true,
            status=0,
            value=NaN,
            gradient=zero(sample0),
            iterations=0,
            objective_evaluations=0,
            hessian_evaluations=0,
            line_search_steps=0,
        )
        return NonlinearResidualUpdate(residual_sample, result)
    end

    transformation_at_point = transformation(lh, expansion_point)
    result = _optimize(
        optimizer,
        sample0;
        fun_and_grad=x -> _nonlinear_residual_value_and_gradient(
            lh,
            expansion_point,
            transformation_at_point,
            metric_sample_array,
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

    if throw_on_failure && _runtime_failure_enabled(sample0) && !result.converged
        throw(
            ErrorException(
                "nonlinear residual update failed with status $(result.status) after $(result.iterations) iterations",
            ),
        )
    end

    return NonlinearResidualUpdate(result.x .- expansion_point, result)
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
    linear_draw = draw_linear_residual(
        lh,
        expansion_point,
        rng;
        draw_linear_kwargs...,
        throw_on_failure=throw_on_failure,
    )

    positive_update = update_nonlinear_residual(
        lh,
        expansion_point,
        linear_draw;
        metric_sample_sign=1,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
        throw_on_failure=throw_on_failure,
    )
    negative_update = update_nonlinear_residual(
        lh,
        expansion_point,
        -linear_draw.residual;
        metric_sample=linear_draw.metric_sample,
        metric_sample_sign=-1,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
        throw_on_failure=throw_on_failure,
    )

    residuals = _stack_residuals(positive_update.residual, negative_update.residual)
    return MirroredResidualDraw(residuals, linear_draw, positive_update, negative_update)
end

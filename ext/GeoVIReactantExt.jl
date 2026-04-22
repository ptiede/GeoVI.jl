module GeoVIReactantExt

import ADTypes
import GeoVI
import LinearAlgebra: dot, norm
import Optimisers
import Random: AbstractRNG
import Reactant
using Reactant: @compile, @jit
using ReactantCore: @trace

const _ReactantArray = Union{Reactant.ConcreteRArray,Reactant.TracedRArray}

struct ReactantVIStepCache{F}
    compiled_step::F
end

function _reactant_optimization_result(
    optimizer;
    x,
    converged,
    status,
    value,
    gradient,
    iterations,
    objective_evaluations,
    hessian_evaluations=0,
    line_search_steps=0,
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
        iterations=iterations,
        objective_evaluations=objective_evaluations,
        hessian_evaluations=hessian_evaluations,
        line_search_steps=line_search_steps,
    )
end

function _forward_to!(y, x, forward)
    copyto!(y, forward(x))
    return nothing
end

struct _ReactantLinearization{F,X,V}
    forward::F
    x::X
    value::V
end

function GeoVI.pushforward(lin::_ReactantLinearization, v::AbstractArray)
    dres, _ = Reactant.Enzyme.autodiff(
        Reactant.Enzyme.ForwardWithPrimal,
        lin.forward,
        Reactant.Enzyme.Duplicated,
        Reactant.Enzyme.Duplicated(lin.x, v),
    )
    return dres
end

function GeoVI.pullback(lin::_ReactantLinearization, η::AbstractArray)
    dx = zero(lin.x)
    dy = copy(η)
    y = zero(lin.value)
    Reactant.Enzyme.autodiff(
        Reactant.Enzyme.Reverse,
        _forward_to!,
        Reactant.Enzyme.Duplicated(y, dy),
        Reactant.Enzyme.Duplicated(lin.x, dx),
        Reactant.Enzyme.Const(lin.forward),
    )
    return dx
end

function _compile_step_vi(
    lh,
    samples::GeoVI.Samples,
    family,
    divergence,
    optimizer,
    state::GeoVI.VIState,
    config::GeoVI.VIConfig,
)
    stripped_state = GeoVI._strip_cache(state)
    return @compile GeoVI._step_vi_impl(
        lh,
        samples,
        family,
        divergence,
        optimizer,
        stripped_state,
        config,
    )
end

function GeoVI._infer_adtype(
    adtype::ADTypes.AutoEnzyme,
    ::Union{Reactant.ConcreteRArray,Reactant.TracedRArray},
)
    return ADTypes.AutoReactant(; mode=adtype)
end

function GeoVI._infer_composed_adtype(
    adtype::ADTypes.AutoEnzyme,
    ::Union{Reactant.ConcreteRArray,Reactant.TracedRArray},
)
    return ADTypes.AutoReactant(; mode=adtype)
end

function GeoVI._infer_adtype(
    ::ADTypes.AutoFiniteDiff,
    ::Union{Reactant.ConcreteRArray,Reactant.TracedRArray},
)
    return ADTypes.AutoReactant()
end

function GeoVI._infer_composed_adtype(
    ::ADTypes.AutoFiniteDiff,
    ::Union{Reactant.ConcreteRArray,Reactant.TracedRArray},
)
    return ADTypes.AutoReactant()
end

function GeoVI._infer_adtype(
    ::ADTypes.NoAutoDiff,
    ::Union{Reactant.ConcreteRArray,Reactant.TracedRArray},
)
    return ADTypes.NoAutoDiff()
end

function GeoVI._infer_composed_adtype(
    ::ADTypes.NoAutoDiff,
    ::Union{Reactant.ConcreteRArray,Reactant.TracedRArray},
)
    return ADTypes.NoAutoDiff()
end

function GeoVI._automatic_linearize(
    ::ADTypes.AutoReactant,
    forward,
    x::AbstractArray;
    fd_eps::Real=1e-6,
)
    return _ReactantLinearization(forward, x, forward(x))
end

function GeoVI._value_and_gradient(
    ::ADTypes.AutoReactant,
    objective,
    x::AbstractArray;
    fd_eps::Real=1e-6,
)
    result = Reactant.Enzyme.gradient(Reactant.Enzyme.ReverseWithPrimal, objective, x)
    return result.val, result.derivs[1]
end

function GeoVI._optimize(
    optimizer::GeoVI.NewtonCG,
    x0::_ReactantArray;
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
        return _reactant_optimization_result(
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

    converged = false
    status = maxiter
    iterations = 0
    active = true

    @trace for iteration in 1:maxiter
        if active
            step, cg_info = GeoVI._conjugate_gradient(
                v -> hessp(x, v),
                grad;
                rtol=cg_rtol,
                atol=cg_atol,
                maxiter=cg_maxiter,
                miniter=cg_miniter,
            )
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

    return _reactant_optimization_result(
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

function GeoVI._optimize(
    optimizer::Optimisers.AbstractRule,
    x0::_ReactantArray;
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

    state = isnothing(optimizer_state) ? @jit(Optimisers.setup(optimizer, x0)) : optimizer_state
    x = x0
    value, grad = fun_and_grad(x)
    objective_evaluations = 1

    if maxiter == 0
        return _reactant_optimization_result(
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

    converged = false
    status = maxiter
    iterations = 0
    active = true

    @trace for iteration in 1:maxiter
        if active
            state, new_x = Optimisers.update(state, x, grad)
            step = new_x .- x
            step_size = stepnorm(step)

            if all(isfinite, new_x) && isfinite(step_size)
                new_value, new_grad = fun_and_grad(new_x)
                objective_evaluations += 1

                if isfinite(new_value)
                    energy_diff = value - new_value
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
                else
                    status = -1
                    iterations = iteration - 1
                    active = false
                end
            else
                status = -1
                iterations = iteration - 1
                active = false
            end
        end
    end

    return _reactant_optimization_result(
        optimizer;
        x=x,
        converged=converged,
        status=status,
        value=value,
        gradient=grad,
        iterations=iterations,
        objective_evaluations=objective_evaluations,
        optimizer_state=state,
    )
end

function GeoVI.draw_linear_residual(
    lh::GeoVI.AbstractLikelihood,
    xi::_ReactantArray,
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
        metric_sample, prior_sample = GeoVI._draw_metric_sample(lh, xi, rng)
    end

    if !from_inverse
        return (
            metric_sample,
            GeoVI._cg_info(converged=true, iterations=0, residual_norm=0.0, breakdown=false),
        )
    end

    operator = v -> GeoVI._posterior_metric(lh, xi, v)
    residual, info = GeoVI._conjugate_gradient(
        operator,
        metric_sample;
        x0=(prior_sample === nothing ? zero(metric_sample) : prior_sample),
        rtol=cg_rtol,
        atol=cg_atol,
        maxiter=cg_maxiter,
        miniter=cg_miniter,
    )

    if throw_on_failure && !Reactant.within_compile() && !info.converged
        throw(
            ErrorException(
                "conjugate gradient failed to converge after $(info.iterations) iterations",
            ),
        )
    end

    return residual, info
end

function GeoVI.update_nonlinear_residual(
    lh::GeoVI.AbstractLikelihood,
    expansion_point::_ReactantArray,
    residual_sample::_ReactantArray;
    rng::Union{Nothing,AbstractRNG}=nothing,
    metric_sample=nothing,
    metric_sample_sign::Number=1,
    optimizer=GeoVI.NewtonCG(),
    optimizer_options=(;),
    throw_on_failure::Bool=true,
)
    if metric_sample === nothing
        rng === nothing &&
            throw(ArgumentError("specify either `rng` or `metric_sample`"))
        metric_sample, _ = GeoVI._draw_metric_sample(lh, expansion_point, rng)
    end

    metric_sample = metric_sample_sign .* metric_sample
    sample0 = expansion_point .+ residual_sample

    maxiter = GeoVI._option(optimizer_options, :maxiter, 20)
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

    transformation_at_point = GeoVI.transformation(lh, expansion_point)
    result = GeoVI._optimize(
        optimizer,
        sample0;
        fun_and_grad=x -> GeoVI._nonlinear_residual_value_and_gradient(
            lh,
            expansion_point,
            transformation_at_point,
            metric_sample,
            x,
        ),
        hessp=(x, v) -> GeoVI._nonlinear_residual_metric(lh, expansion_point, x, v),
        maxiter=maxiter,
        miniter=GeoVI._option(optimizer_options, :miniter, 0),
        xtol=GeoVI._option(optimizer_options, :xtol, 1e-5),
        absdelta=GeoVI._option(optimizer_options, :absdelta, nothing),
        cg_rtol=GeoVI._option(optimizer_options, :cg_rtol, 1e-8),
        cg_atol=GeoVI._option(optimizer_options, :cg_atol, 0.0),
        cg_maxiter=GeoVI._option(optimizer_options, :cg_maxiter, nothing),
        cg_miniter=GeoVI._option(optimizer_options, :cg_miniter, 0),
        stepnorm=step -> GeoVI._nonlinear_residual_stepnorm(lh, expansion_point, step),
    )

    if throw_on_failure && !Reactant.within_compile() && !result.converged
        throw(
            ErrorException(
                "nonlinear residual update failed with status $(result.status) after $(result.iterations) iterations",
            ),
        )
    end

    return result.x .- expansion_point, result
end

function GeoVI._step_vi(
    adtype::ADTypes.AutoReactant,
    lh::GeoVI.AbstractLikelihood,
    samples::GeoVI.Samples,
    family::GeoVI.AbstractVariationalFamily,
    divergence::GeoVI.AbstractDivergence,
    optimizer,
    state::GeoVI.VIState,
    config::GeoVI.VIConfig,
)
    state.rng isa Reactant.ReactantRNG ||
        return GeoVI._step_vi_default(
            adtype,
            lh,
            samples,
            family,
            divergence,
            optimizer,
            state,
            config,
        )

    cache = if state.cache isa ReactantVIStepCache
        state.cache
    else
        ReactantVIStepCache(
            _compile_step_vi(lh, samples, family, divergence, optimizer, state, config),
        )
    end

    new_samples, new_state = cache.compiled_step(
        lh,
        samples,
        family,
        divergence,
        optimizer,
        GeoVI._strip_cache(state),
        config,
    )
    return new_samples, GeoVI._restore_cache(new_state, cache)
end

end

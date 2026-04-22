module GeoVIReactantExt

import ADTypes
import GeoVI
import GeoVI: _check_objective_value, _outer_vi_value_and_gradient, _run_optimizer_rule
import LinearAlgebra: dot, norm
import Optimisers
import Random: AbstractRNG
import Reactant
using Reactant: @compile, @jit
using ReactantCore: @trace

const _ReactantArray = Union{Reactant.ConcreteRArray,Reactant.TracedRArray}

function Reactant.traced_type_inner(
    ::Type{GeoVI.ConjugateGradient},
    seen,
    ::Reactant.TraceMode,
    ::Type,
    ndevices,
    runtime,
)
    return GeoVI.ConjugateGradient
end

function Reactant.make_tracer(
    seen,
    prev::GeoVI.ConjugateGradient,
    path,
    mode;
    kwargs...,
)
    return prev
end

mutable struct ReactantVIStepCache
    compiled_step::Any
    signature::Any
end

struct ReactantOptimizerState{R,S}
    rule::R
    state::S
end

_optimizer_leaf(state::ReactantOptimizerState) = Optimisers.Leaf(state.rule, state.state, false)

function _reactant_optimizer_state(state::Optimisers.Leaf)
    return ReactantOptimizerState(state.rule, state.state)
end

function _reactant_optimizer_state(state::ReactantOptimizerState)
    return state
end

function GeoVI._optimizer_state(
    optimizer::Optimisers.AbstractRule,
    x0,
    previous_state::ReactantOptimizerState,
)
    return previous_state
end

function GeoVI._prepare_optimizer_state(
    optimizer::Optimisers.AbstractRule,
    x0::_ReactantArray,
    optimizer_state,
)
    if isnothing(optimizer_state)
        return _reactant_optimizer_state(@jit(Optimisers.setup(optimizer, x0)))
    end
    return _reactant_optimizer_state(optimizer_state)
end

function GeoVI._optimizer_update(
    state::ReactantOptimizerState,
    x,
    grad,
)
    leaf_state = _optimizer_leaf(state)
    leaf_state, new_x = Optimisers.update(leaf_state, x, grad)
    return _reactant_optimizer_state(leaf_state), new_x
end

GeoVI._runtime_failure_enabled(::_ReactantArray) = !Reactant.within_compile()

function _reactant_strip_state(state::GeoVI.VIState)
    minimization_state = if state.minimization_state isa GeoVI.OptimizationResult
        state.minimization_state.optimizer_state
    else
        state.minimization_state
    end
    return GeoVI.VIState(
        iteration=state.iteration,
        rng=state.rng,
        sample_state=state.sample_state,
        minimization_state=minimization_state,
    )
end

function _evaluate_outer_vi_candidate(
    step_valid,
    value,
    grad,
    new_x,
    objective_evaluations,
    objective_state,
)
    adtype, divergence, likelihood, residuals, fd_eps = objective_state
    new_value = value
    new_grad = grad
    value_valid = false
    evaluations = objective_evaluations
    @trace if step_valid
        new_value, new_grad = _outer_vi_value_and_gradient(
            adtype,
            divergence,
            likelihood,
            residuals,
            new_x;
            fd_eps=fd_eps,
        )
        evaluations += 1
        value_valid = _check_objective_value(new_value)
    end
    return new_value, new_grad, value_valid, evaluations
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
    problem::GeoVI.VariationalProblem,
    samples::GeoVI.Samples,
    state::GeoVI.VIState,
)
    stripped_state = _reactant_strip_state(state)
    return @compile GeoVI._step_vi_impl(problem, samples, stripped_state)
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
    fd_eps=1e-6,
)
    return _ReactantLinearization(forward, x, forward(x))
end

function GeoVI._value_and_gradient(
    ::ADTypes.AutoReactant,
    objective,
    x::AbstractArray;
    fd_eps=1e-6,
)
    result = Reactant.Enzyme.gradient(Reactant.Enzyme.ReverseWithPrimal, objective, x)
    return result.val, result.derivs[1]
end

GeoVI._materialize_step_position(x::_ReactantArray) = identity.(x)

function GeoVI._update_position(
    problem::GeoVI.VariationalProblem{L,S,F,D,O,C,AD,DL,NU,OO},
    samples::GeoVI.Samples,
    previous_minimization_state=nothing,
) where {L,S,F,D,O<:Optimisers.AbstractRule,C,AD<:ADTypes.AutoReactant,DL,NU,OO}
    optimizer_state = GeoVI._previous_optimizer_state(
        problem.optimizer,
        samples.position,
        previous_minimization_state,
    )
    state = GeoVI._prepare_optimizer_state(
        problem.optimizer,
        samples.position,
        optimizer_state,
    )
    value, grad = _outer_vi_value_and_gradient(
        problem.adtype,
        problem.divergence,
        problem.likelihood,
        samples.residuals,
        samples.position;
        fd_eps=problem.optimizer_options.fd_eps,
    )
    return _run_optimizer_rule(
        problem.optimizer,
        state,
        samples.position,
        value,
        grad;
        maxiter=problem.optimizer_options.maxiter,
        miniter=problem.optimizer_options.miniter,
        xtol=problem.optimizer_options.xtol,
        absdelta=problem.optimizer_options.absdelta,
        stepnorm=norm,
        objective_evaluations=1,
        evaluate_candidate=_evaluate_outer_vi_candidate,
        evaluation_state=(
            problem.adtype,
            problem.divergence,
            problem.likelihood,
            samples.residuals,
            problem.optimizer_options.fd_eps,
        ),
    )
end

function GeoVI._step_vi(
    adtype::ADTypes.AutoReactant,
    problem::GeoVI.VariationalProblem,
    samples::GeoVI.Samples,
    state::GeoVI.VIState,
)
    state.rng isa Reactant.ReactantRNG ||
        return GeoVI._step_vi_default(problem, samples, state)

    stripped_state = _reactant_strip_state(state)
    signature = (typeof(problem), typeof(samples), typeof(stripped_state))
    cache = if state.cache isa ReactantVIStepCache
        state.cache
    else
        ReactantVIStepCache(nothing, nothing)
    end

    if cache.compiled_step === nothing || cache.signature != signature
        cache.compiled_step = _compile_step_vi(problem, samples, state)
        cache.signature = signature
    end

    new_samples, new_state = cache.compiled_step(
        problem,
        samples,
        stripped_state,
    )
    return new_samples, GeoVI._restore_cache(new_state, cache)
end

end

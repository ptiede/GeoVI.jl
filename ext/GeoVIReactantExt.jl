module GeoVIReactantExt

import ADTypes
import GeoVI
import LinearAlgebra: dot, norm
import Optimisers
import Random: AbstractRNG
import Reactant
using Reactant: @compile, @jit
using ReactantCore: @trace

const _ReactantArray = Union{Reactant.ConcreteRArray, Reactant.TracedRArray}

# Note: the family/optimizer/estimator/ConjugateGradient config structs need no
# `make_tracer`/`traced_type_inner` pass-through hooks. They flow into the
# compiled `step_vi!` as arguments, but Reactant treats their plain
# Int/Float64/Bool fields as compile-time constants, and the hot loops use
# `@trace ... track_numbers=false` so the deep number-walk is suppressed there.

struct ReactantOptimizerState{R, S}
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

function _forward_to!(y, x, forward)
    copyto!(y, forward(x))
    return nothing
end

struct _ReactantLinearization{F, X, V}
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

function GeoVI._infer_adtype(
        adtype::ADTypes.AutoEnzyme,
        ::Union{Reactant.ConcreteRArray, Reactant.TracedRArray},
    )
    return ADTypes.AutoReactant(; mode = adtype)
end

function GeoVI._infer_composed_adtype(
        adtype::ADTypes.AutoEnzyme,
        ::Union{Reactant.ConcreteRArray, Reactant.TracedRArray},
    )
    return ADTypes.AutoReactant(; mode = adtype)
end

function GeoVI._infer_adtype(
        ::ADTypes.AutoFiniteDiff,
        ::Union{Reactant.ConcreteRArray, Reactant.TracedRArray},
    )
    return ADTypes.AutoReactant()
end

function GeoVI._infer_composed_adtype(
        ::ADTypes.AutoFiniteDiff,
        ::Union{Reactant.ConcreteRArray, Reactant.TracedRArray},
    )
    return ADTypes.AutoReactant()
end

function GeoVI._infer_adtype(
        ::ADTypes.NoAutoDiff,
        ::Union{Reactant.ConcreteRArray, Reactant.TracedRArray},
    )
    return ADTypes.NoAutoDiff()
end

function GeoVI._infer_composed_adtype(
        ::ADTypes.NoAutoDiff,
        ::Union{Reactant.ConcreteRArray, Reactant.TracedRArray},
    )
    return ADTypes.NoAutoDiff()
end

function GeoVI._automatic_linearize(
        ::ADTypes.AutoReactant,
        forward,
        x::AbstractArray;
        fd_eps = 1.0e-6,
    )
    return _ReactantLinearization(forward, x, forward(x))
end

function GeoVI._value_and_gradient(
        ::ADTypes.AutoReactant,
        objective,
        x::AbstractArray;
        fd_eps = 1.0e-6,
    )
    result = Reactant.Enzyme.gradient(Reactant.Enzyme.ReverseWithPrimal, objective, x)
    return result.val, result.derivs[1]
end

GeoVI._wrap_rng(::ADTypes.AutoReactant, rng::Reactant.ReactantRNG) = rng
function GeoVI._wrap_rng(::ADTypes.AutoReactant, rng::AbstractRNG)
    seed = rand(rng, UInt64, 2)
    return Reactant.ReactantRNG(Reactant.to_rarray(seed))
end

# Size the white-noise buffer under `@jit`: the tangent template runs the forward
# model (e.g. a matmul on `ConcretePJRTArray`), which cannot execute eagerly.
GeoVI._tangent_template(::ADTypes.AutoReactant, lh::GeoVI.AbstractLikelihood, xi) =
    @jit(GeoVI._metric_tangent_template(lh, xi))

# Drive the loop under Reactant: compile `step_vi!` ONCE for the preallocated
# buffers (with `n_refine` as a `ConcreteRNumber` so its inner refinement loop is
# a runtime bound — one compile serves any count), then call the compiled thunk
# `n_iterations` times. `step_vi!` mutates the state's arrays in place, so the
# loop just re-invokes it. Users wanting a custom loop call `@compile step_vi!`
# themselves (the same primitive).
function GeoVI._run_vi!(
        ::ADTypes.AutoReactant, rng, problem::GeoVI.VariationalProblem, state::GeoVI.VIState,
        n_iterations, n_refine,
    )
    # `init` always wraps the rng into a `Reactant.ReactantRNG` for an AutoReactant
    # problem, so by here `rng` is device-side and its draws advance per compiled
    # call. (A host rng would be baked in as a compile-time constant — frozen noise.)
    nref = Reactant.ConcreteRNumber(Int(n_refine))
    @info "GeoVI: compiling step_vi!..."
    t_compile = @elapsed begin
        cstep = @compile GeoVI.step_vi!(rng, problem, state, nref)
    end
    @info "GeoVI: compilation done in" t_compile
    for _ in 1:n_iterations
        cstep(rng, problem, state, nref)
    end
    return state
end

end

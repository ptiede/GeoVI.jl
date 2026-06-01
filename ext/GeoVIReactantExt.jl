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

# A frozen-flag-free image of an Optimisers `Leaf` so Reactant does not trace the
# `frozen::Bool`. An optimizer-state *tree* (a single `Leaf` for a bare-array θ, or
# a NamedTuple/Tuple of `Leaf`s for a structured θ such as mean-field's
# `(; mean, logstd)`) is mapped leaf-wise to the same structure of `ReactantLeaf`s.
struct ReactantLeaf{R, S}
    rule::R
    state::S
end

struct ReactantOptimizerState{T}
    tree::T
end

# Optimisers state tree → ReactantOptimizerState (strip the per-leaf frozen flag).
_to_reactant_state(state::ReactantOptimizerState) = state
_to_reactant_state(state) = ReactantOptimizerState(
    GeoVI.fmap(l -> ReactantLeaf(l.rule, l.state), state; exclude = x -> x isa Optimisers.Leaf),
)
# ReactantOptimizerState → Optimisers state tree (rebuild `Leaf`s, frozen = false).
_from_reactant_state(s::ReactantOptimizerState) = GeoVI.fmap(
    rl -> Optimisers.Leaf(rl.rule, rl.state, false), s.tree; exclude = x -> x isa ReactantLeaf,
)

# True if `x` (or any of its leaves) is a Reactant array.
_any_reactant(x::_ReactantArray) = true
_any_reactant(x::AbstractArray) = false
_any_reactant(x) = any(_any_reactant, Optimisers.trainables(x))

# Build (and wrap) the optimizer state at `init` for a Reactant θ, so the value
# stored in the `VIState` — and thus the `@compile step_vi!` argument — is already
# frozen-stripped. Covers a bare-array θ and a structured (NamedTuple/Tuple) θ
# whose leaves are Reactant arrays; defers to the generic (eager) path otherwise.
function GeoVI._optimizer_state(
        optimizer::Optimisers.AbstractRule,
        x0::Union{_ReactantArray, NamedTuple, Tuple},
        previous_state,
    )
    previous_state isa ReactantOptimizerState && return previous_state
    _any_reactant(x0) || return invoke(
        GeoVI._optimizer_state,
        Tuple{Optimisers.AbstractRule, Any, Any},
        optimizer, x0, previous_state,
    )
    if previous_state isa GeoVI.OptimizationResult &&
            previous_state.optimizer == optimizer &&
            previous_state.optimizer_state !== nothing
        return previous_state.optimizer_state
    end
    return _to_reactant_state(@jit(Optimisers.setup(optimizer, x0)))
end

# Inside the compiled `_optimize`, `init` has already produced the wrapped state
# (via `_optimizer_state` below), so this is a pure type-dispatch pass-through with
# no runtime branching — branching on a traced value here is a tracing error.
GeoVI._prepare_optimizer_state(
    optimizer::Optimisers.AbstractRule,
    x0,
    optimizer_state::ReactantOptimizerState,
) = optimizer_state

function GeoVI._optimizer_update(
        state::ReactantOptimizerState,
        x,
        grad,
    )
    opt_tree = _from_reactant_state(state)
    new_tree, new_x = Optimisers.update(opt_tree, x, grad)
    return _to_reactant_state(new_tree), new_x
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
        x;
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

# Drive the loop under Reactant: compile `step_vi!` ONCE for the preallocated
# buffers, then call the compiled thunk `n_iterations` times. `step_vi!` mutates the
# state's arrays in place, so the loop just re-invokes it. The family's `draw_samples!`
# keeps its `randn` outside the traced loop (rng never enters `@trace`), and the metric
# tangent template is built inside the compiled step, so the whole draw traces normally.
# Users wanting a custom loop call `@compile step_vi!` themselves (the same primitive).
function GeoVI._run_vi!(
        ::ADTypes.AutoReactant, rng, problem::GeoVI.VariationalProblem, state::GeoVI.VIState,
        n_iterations,
    )
    # `init` always wraps the rng into a `Reactant.ReactantRNG` for an AutoReactant
    # problem, so by here `rng` is device-side and its draws advance per compiled
    # call. (A host rng would be baked in as a compile-time constant — frozen noise.)
    @info "GeoVI: compiling step_vi!..."
    t_compile = @elapsed begin
        cstep = @compile GeoVI.step_vi!(rng, problem, state)
    end
    @info "GeoVI: compilation done in" t_compile
    for _ in 1:n_iterations
        cstep(rng, problem, state)
    end
    return state
end

end

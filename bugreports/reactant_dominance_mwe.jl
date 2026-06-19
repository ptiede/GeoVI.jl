# Minimal reproducer: Reactant XLA compilation fails with
#   error: operand #N does not dominate this use
# (an SSA-dominance failure at `compile_mlir!`; tracing itself succeeds).
#
# Trigger — all of these are required (removing any one makes it compile):
#   * a `@trace while` loop;
#   * a `@trace if` that conditionally calls an Enzyme value+gradient
#     (`ReverseWithPrimal`), whose result is consumed by a SECOND `@trace if`;
#   * the differentiated function reduces over a TRACED array input `R`
#     (making `R` a compile-time constant instead does not trigger it);
#   * the primal value is kept (returning only the gradient compiles).
#
# deps: Reactant (+ ReactantCore, Reactant.Enzyme)
using Reactant
using ReactantCore: @trace, within_compile, promote_to_traced
const Enzyme = Reactant.Enzyme

# scalar function of a NamedTuple θ, reducing over the rows of a TRACED matrix R.
# (A plain unrolled `for` reproduces identically; `@trace for` matches the real code.)
function loss(θ, R)
    s = zero(eltype(θ.mean))
    @trace track_numbers = false for i in 1:4
        s += 0.5f0 * sum(abs2, θ.mean .+ exp.(θ.logstd) .* R[i, :])
    end
    return s / 4 - sum(θ.logstd)
end
val_and_grad(θ, R) = (r = Enzyme.gradient(Enzyme.ReverseWithPrimal, Base.Fix2(loss, R), θ); (r.val, r.derivs[1]))

# @trace if #1: conditionally evaluate value+gradient (an autodiff call).
function maybe_grad(ok, g, θ, vg)
    @trace if ok
        _, gout = vg(θ)
        copyto!(g, gout)
    end
    return g
end
# @trace if #2: conditionally consume that gradient.
function maybe_step(ok, θ, g, newθ, newg)
    @trace if ok
        θ, g = newθ, newg
    end
    return θ, g
end

function go(θ, R)
    vg = Base.Fix2(val_and_grad, R)
    _, g = vg(θ)
    i = 0
    within_compile() && (i = promote_to_traced(i))
    @trace track_numbers = false while i < 1
        newθ = (; mean = θ.mean .- 0.05f0 .* g.mean, logstd = θ.logstd .- 0.05f0 .* g.logstd)
        ok = all(isfinite, newθ.mean) & all(isfinite, newθ.logstd)
        newg = maybe_grad(ok, g, newθ, vg)
        θ, g = maybe_step(ok, θ, g, newθ, newg)
        i += 1
    end
    return θ
end

θ = (; mean = Reactant.to_rarray(zeros(Float32, 3)), logstd = Reactant.to_rarray(zeros(Float32, 3)))
R = Reactant.to_rarray(Float32[0.5 -0.3 0.1; -0.2 0.4 0.0; 0.1 0.1 -0.5; 0.3 -0.1 0.2])
Reactant.@compile go(θ, R)

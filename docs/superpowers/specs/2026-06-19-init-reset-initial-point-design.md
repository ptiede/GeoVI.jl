# Resetting the initial latent point: `init(…, ξ0)` and `reset!`

## Problem

To restart VI from a different initial latent point `ξ₀`, the user currently has to
rebuild the entire `VariationalProblem` (the point is baked into
`problem.initial_samples.position`). That is annoying for the common
"try a different start" workflow, and it is wasteful under Reactant: re-running
`init` allocates new buffers, forcing a recompile of any `step_vi!` thunk.

## Design

Two complementary entry points, mirroring `init` (allocating) vs an in-place
reset. They share the existing `init` machinery — `init_params(family, latent)`
to build `θ` and the latent point to size buffers.

### A. `init` gains an optional latent point (allocates)

```julia
init(rng, problem, ξ0)        # rng-first positional (API convention)
init(problem, ξ0; rng=...)
init(rng, problem)            # unchanged
init(problem; rng=...)        # unchanged
```

The only change in the body is selecting the latent point:

```julia
latent = something(ξ0, problem.initial_samples.position)
θ       = init_params(problem.family, latent)
```

Everything downstream (residual buffer sized from `latent`, optimizer state from
`θ`, rng wrapping) is unchanged. `ξ0` defaults to `nothing`, so existing call
sites are untouched.

`ξ0` is the **latent point** (a flat white array, same kind as the user's
original `ξ₀` / `problem.initial_samples.position`), **not** a `θ` parameter
container. For mean-field this means the caller still passes a latent array;
`init_params` turns it into `(; mean, logstd)`.

### B. `reset!` re-initializes an existing `VIState` in place

```julia
reset!(state, problem, ξ0)  ->  state
```

Definition: **`reset!` is `init` that reuses `state`'s buffers instead of
allocating.** It performs a full logical restart while preserving every array
identity:

1. `θ_new = init_params(problem.family, ξ0)`; write into `state.position` via
   `fmap(copyto!, state.position, θ_new)` (handles both the bare-array `θ` and
   the mean-field NamedTuple `θ`).
2. `fill!(state.residuals, 0)` when `state.residuals !== nothing` (matches a
   fresh `init`; the buffer is redrawn at the start of every `step_vi!` anyway).
3. Reset the optimizer state to a fresh `_init_optimizer_state(problem, state.position)`,
   copied into the existing `state.optimizer_state` in place
   (`fmap(copyto!, …)`) so array identity is preserved. `NewtonCG`'s state is
   `nothing`, so that case is a no-op.

`reset!` returns `state`.

**Semantics confirmed with the user:**

- **Optimizer state is reset** (fresh momentum), not preserved — "overwrite the
  initial point" reads as a restart; stale Adam momentum from a different point
  would be more surprising than helpful. Buffer identity is preserved
  independently (the values are zeroed in place), so Reactant compilation is
  unaffected.
- **`reset!` does not touch or return the rng.** The rng was wrapped and is owned
  by the user from the original `init`; `reset!` returns only `state`.

**Size requirement:** in-place reuse cannot resize. `ξ0` must match the original
latent's size/shape. A mismatch is a documented error directing the user to the
allocating `init` instead. (`copyto!` already throws on a shape mismatch; we add
a clear error message / size check up front.)

## Reactant interaction

The whole point of `reset!` over a fresh `init`: every buffer keeps its identity,
so a previously `@compile`d `step_vi!(rng, problem, state)` thunk stays valid and
is reused. No host-only state is introduced; `reset!` is a plain host-side
mutation (the `fmap(copyto!, …)` / `fill!` calls), run once between optimization
runs, not inside the compiled step.

## Exports

Add `reset!` to the VI-loop export block in `src/GeoVI.jl` (next to `init`,
`step_vi!`, `fit`). `init`'s new method needs no export change.

## Testing

- `init(rng, problem, ξ0)` produces a state whose `position` derives from `ξ0`,
  not `problem.initial_samples.position` (check for both a Fisher-Gaussian family
  and `MeanFieldGaussian`, where `θ` is a NamedTuple).
- `init(rng, problem)` (no `ξ0`) is unchanged — equal to the previous behavior.
- `reset!(state, problem, ξ0)` mutates `state` in place: returns the same object,
  `state.position` derives from `ξ0`, residuals are zeroed, optimizer state is
  fresh. Assert array identities (`===`) of `state.residuals` and the optimizer
  state's leaves are preserved across the call.
- `reset!` with a wrong-sized `ξ0` errors clearly.
- A short end-to-end check: fit, `reset!` to a new point, fit again, and confirm
  the second run starts from the reset point.
- (If Reactant is available in CI) compile `step_vi!`, `reset!`, and confirm the
  compiled thunk still runs — otherwise covered by the identity-preservation
  assertions.

## Out of scope

- Preserving optimizer momentum across a reset (explicitly rejected above).
- Resizing buffers in `reset!` (use allocating `init`).
- Any change to `step_vi!`, `fit`, or the four-axis problem construction.

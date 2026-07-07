# init(ξ0) / reset! Initial-Point Override — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users restart VI from a new initial latent point without rebuilding the `VariationalProblem` — an optional `ξ0` to `init` (allocating) and a new `reset!` (in-place), with `init`/`fit` rng moved from keyword to first-positional-or-auto.

**Architecture:** `init` already derives `θ` via `init_params(family, latent)` and sizes buffers from `latent`. We add an optional `ξ0` that overrides `problem.initial_samples.position`, and a `reset!(state, problem, ξ0)` that re-runs that logic into an existing `VIState`'s buffers in place (preserving `position`/`residuals` array identity for Reactant, reassigning the optimizer state exactly as `update!` does). Separately, `init` and `fit` drop their `; rng=…` keyword form in favor of `f(rng, …)` (positional) or `f(…)` (auto `Random.default_rng()`).

**Tech Stack:** Julia, Functors (`fmap`), Optimisers, Reactant (extension; not required to test the core).

## Global Constraints

- The rng is **first-positional or auto-generated, never a keyword** (project convention).
- `ξ0` is the **latent point** (a flat white array like `problem.initial_samples.position`), NOT a `θ` parameter container. `init_params(family, ξ0)` converts it (identity-copy for Fisher-Gaussian, `(; mean=ξ0, logstd=0)` for mean-field).
- `reset!` preserves the array identity of `state.position` and `state.residuals` (Reactant in-place aliasing); it **reassigns** `state.optimizer_state` (matching `update!`, vi.jl:270 — `fmap(copyto!)` cannot be used on optimizer trees because of non-array leaves like Adam's `βt`).
- `reset!` cannot resize: `ξ0` must match the existing latent size; mismatch throws `DimensionMismatch` with a message pointing to `init`.
- Existing module conventions: `fmap` is imported (`using Functors: fmap`, GeoVI.jl:4); `Random` and `Optimisers` are in scope in `src/vi.jl`.
- Run the test suite with: `julia --project -e 'using Pkg; Pkg.test()'`. A single testset can be exercised from the REPL via `include("test/runtests.jl")`.

---

### Task 1: `init` — optional `ξ0`, rng positional-or-auto

**Files:**
- Modify: `src/vi.jl:144-169` (the `init` docstring + the two `init` methods)
- Test: `test/runtests.jl` (add a `@testset "init/reset! initial point"` block; see Step 1 for placement)

**Interfaces:**
- Consumes: `init_params(problem.family, latent)`, `_init_residual_buffer(problem, latent)`, `_init_optimizer_state(problem, θ)`, `_wrap_rng(problem.adtype, rng)`, `VIState(θ, residuals, optimizer_state)` — all existing in `src/vi.jl`.
- Produces:
  - `init(rng::AbstractRNG, problem::VariationalProblem, ξ0=nothing) -> (wrapped_rng, VIState)`
  - `init(problem::VariationalProblem) -> (wrapped_rng, VIState)` (auto rng)
  - `init(problem::VariationalProblem, ξ0) -> (wrapped_rng, VIState)` (auto rng)
  - The `init(problem; rng=…)` keyword method is **removed**.

- [ ] **Step 1: Write the failing test**

Add this block at the end of the `@testset "outer VI loop"` body in `test/runtests.jl` (immediately before the `end` that closes that testset — locate it by the comment `# outer VI loop` near line 571). It reuses the `mgvi_problem` and `est`/`outer` already built in that testset:

```julia
@testset "init/reset! initial point" begin
    # ── init with an explicit ξ0 overrides problem.initial_samples.position ──
    xi_new = [3.0]
    rng_a, st_a = init(MersenneTwister(5), mgvi_problem, xi_new)
    @test st_a.position == xi_new          # θ derived from ξ0 (array family: θ is the latent)
    @test st_a.position !== xi_new         # …but a fresh copy, not aliased
    @test size(st_a.residuals) == (8, 1)   # buffer still sized from the (same-length) latent

    # ── init with no ξ0 is unchanged: derives from the problem's xi0 ──
    _, st_b = init(MersenneTwister(5), mgvi_problem)
    @test st_b.position == xi0

    # ── auto-rng forms run and match the problem's xi0 ──
    _, st_c = init(mgvi_problem)
    @test st_c.position == xi0
    _, st_d = init(mgvi_problem, xi_new)
    @test st_d.position == xi_new

    # ── mean-field: ξ0 seeds θ.mean (θ is a NamedTuple) ──
    mf_problem = VariationalProblem(
        lh, xi0;
        family = MeanFieldGaussian(),
        estimator = MCEstimator(n_samples = 4, mirrored = true),
        optimizer = Optimisers.Adam(0.05),
    )
    _, st_mf = init(MersenneTwister(5), mf_problem, xi_new)
    @test st_mf.position.mean == xi_new
    @test all(iszero, st_mf.position.logstd)
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project -e 'include("test/runtests.jl")'`
Expected: FAIL — `init(::MersenneTwister, ::VariationalProblem, ::Vector)` has no method (3-arg form not defined yet).

- [ ] **Step 3: Replace the `init` methods and docstring**

In `src/vi.jl`, replace the docstring + both methods (currently lines 144-169) with:

```julia
"""
    init([rng], problem[, ξ0]) -> (rng, state)

Allocate the [`VIState`](@ref) for `problem` — a fresh copy of the initial
position, the residual buffer, and the optimizer state — and return it together
with the loop RNG to thread through [`step_vi!`](@ref).

`ξ0` optionally overrides the problem's initial latent point
(`problem.initial_samples.position`); it is the **starting latent point** (for
MGVI/geoVI the mean the optimizer starts from, for mean-field the seed for
`(; mean = ξ0, logstd = 0)`). This is how you restart from a new point without
rebuilding the `problem`. To restart an *existing* state in place (reusing its
buffers), use [`reset!`](@ref) instead.

`rng` is first-positional or auto-generated (`Random.default_rng()`); it is never
a keyword. For an `AutoReactant` problem the RNG is wrapped into a
`Reactant.ReactantRNG` (the compiled step is built lazily by `fit`, or by the
user calling `@compile step_vi!(...)`).

The residual buffer starts zero-filled; [`step_vi!`](@ref) redraws it at the
start of every step.
"""
function init(rng::AbstractRNG, problem::VariationalProblem, ξ0 = nothing)
    # All buffers are sized from the latent point ξ₀ (latent-shaped by
    # construction), so the interface needs no `θ → latent` projection.
    latent = something(ξ0, problem.initial_samples.position)
    θ = init_params(problem.family, latent)
    residuals = _init_residual_buffer(problem, latent)
    optimizer_state = _init_optimizer_state(problem, θ)
    wrapped_rng = _wrap_rng(problem.adtype, rng)
    return wrapped_rng, VIState(θ, residuals, optimizer_state)
end

init(problem::VariationalProblem) = init(Random.default_rng(), problem)
init(problem::VariationalProblem, ξ0) = init(Random.default_rng(), problem, ξ0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project -e 'include("test/runtests.jl")'`
Expected: PASS for the new `init/reset! initial point` testset (the `reset!` sub-tests are added in Task 2; the `init` assertions above pass now).

- [ ] **Step 5: Commit**

```bash
git add src/vi.jl test/runtests.jl
git commit -m "feat: init gains optional ξ0 starting point; rng positional-or-auto"
```

---

### Task 2: `reset!` — in-place re-initialization

**Files:**
- Modify: `src/vi.jl` (add `reset!` and a small `_latent_ref` helper just after the `init` methods)
- Modify: `src/GeoVI.jl:54` (export `reset!` in the VI-loop export block)
- Test: `test/runtests.jl` (extend the `@testset "init/reset! initial point"` from Task 1)

**Interfaces:**
- Consumes: `init_params(problem.family, ξ0)`, `_init_optimizer_state(problem, state.position)`, `fmap` (Functors), `VIState` fields `position`/`residuals`/`optimizer_state`.
- Produces: `reset!(state::VIState, problem::VariationalProblem, ξ0::AbstractArray) -> state`. Also internal `_latent_ref(θ)` returning the latent-sized reference array for a `θ` container.

- [ ] **Step 1: Write the failing test**

Append these tests inside the `@testset "init/reset! initial point"` block (after the mean-field assertions from Task 1, before its closing `end`):

```julia
    # ── reset! re-initializes an existing state in place ──
    rng_r, st = init(MersenneTwister(5), mgvi_problem)         # NewtonCG ⇒ optimizer_state === nothing
    pos_before = st.position
    res_before = st.residuals
    fill!(st.residuals, 7.0)                                   # dirty the residual buffer
    out = reset!(st, mgvi_problem, xi_new)
    @test out === st                                           # returns the same object
    @test st.position === pos_before                           # position identity preserved (Reactant)
    @test st.residuals === res_before                          # residual identity preserved (Reactant)
    @test st.position == xi_new                                # …but values now derive from ξ0
    @test all(iszero, st.residuals)                            # residuals zeroed
    @test st.optimizer_state === nothing                       # NewtonCG: still nothing

    # ── reset! with a stateful optimizer reassigns a fresh (zeroed) state ──
    rng_o, st_o = init(MersenneTwister(5), mf_problem)
    for _ in 1:5
        step_vi!(rng_o, mf_problem, st_o)                      # build Adam momentum
    end
    @test st_o.optimizer_state !== nothing
    reset!(st_o, mf_problem, xi_new)
    @test st_o.position.mean == xi_new
    @test all(iszero, st_o.position.logstd)
    # optimizer state is fresh: flattened state equals a freshly-init'd one at the
    # same point (destructure flattens the whole Leaf tree, incl. zeroed momentum).
    _, st_fresh = init(MersenneTwister(5), mf_problem, xi_new)
    @test Optimisers.destructure(st_o.optimizer_state)[1] ==
        Optimisers.destructure(st_fresh.optimizer_state)[1]

    # ── reset! cannot resize: wrong-length ξ0 errors clearly ──
    @test_throws DimensionMismatch reset!(st, mgvi_problem, [1.0, 2.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project -e 'include("test/runtests.jl")'`
Expected: FAIL — `reset!` is not defined (`UndefVarError: reset!`).

- [ ] **Step 3: Implement `reset!`**

In `src/vi.jl`, immediately after the new `init(problem, ξ0)` method from Task 1, add:

```julia
# Latent-sized reference array inside a θ container, for reset!'s size check.
# Array families: θ *is* the latent. Mean-field: θ.mean is latent-sized.
_latent_ref(θ::AbstractArray) = θ
_latent_ref(θ) = first(values(θ))

"""
    reset!(state, problem, ξ0) -> state

Re-initialize an existing [`VIState`](@ref) in place to restart from a new
starting latent point `ξ0`, reusing `state`'s buffers instead of allocating.
This is [`init`](@ref) for an already-allocated state: the variational parameters
are rebuilt from `ξ0` via `init_params`, the residual buffer is zeroed (it is
redrawn at the start of every [`step_vi!`](@ref) anyway), and the optimizer state
is reset to fresh (momentum cleared).

`state.position` and `state.residuals` keep their array identity (so a compiled
`step_vi!` thunk stays valid under Reactant); the optimizer state is reassigned,
matching [`update!`](@ref)'s per-step behavior. `ξ0` must match the existing
latent size — `reset!` cannot resize; call [`init`](@ref) for a different size.
"""
function reset!(state::VIState, problem::VariationalProblem, ξ0::AbstractArray)
    ref = _latent_ref(state.position)
    size(ref) == size(ξ0) || throw(
        DimensionMismatch(
            "reset! cannot resize: state latent size $(size(ref)), new ξ0 size " *
                "$(size(ξ0)). Use `init` to allocate a fresh state of the new size."
        ),
    )
    θ_new = init_params(problem.family, ξ0)
    # Leaf-wise in-place copy preserves position-buffer identity (cf. `update!`);
    # for a bare-array θ this is exactly `copyto!(state.position, θ_new)`.
    fmap(copyto!, state.position, θ_new)
    state.residuals === nothing ||
        fill!(state.residuals, zero(eltype(state.residuals)))
    # Reassign (do NOT fmap-copy): optimizer trees have non-array leaves.
    state.optimizer_state = _init_optimizer_state(problem, state.position)
    return state
end
```

- [ ] **Step 4: Export `reset!`**

In `src/GeoVI.jl`, the VI-loop export block (lines 50-61) lists `init,` then later `step_vi!,`/`fit`. Add `reset!,` on the line after `init,`:

```julia
    init,
    reset!,
    draw_metric_sample,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `julia --project -e 'include("test/runtests.jl")'`
Expected: PASS for the full `init/reset! initial point` testset.

- [ ] **Step 6: Commit**

```bash
git add src/vi.jl src/GeoVI.jl test/runtests.jl
git commit -m "feat: reset! re-initializes a VIState in place from a new ξ0"
```

---

### Task 3: `fit` — drop the rng keyword

**Files:**
- Modify: `src/vi.jl:524-532` (the `fit` docstring region + the keyword method)
- Modify: `test/runtests.jl` — every `fit(p, n; rng = R)` call site (lines ~310, 596, 628, 642, 713, 778, 808, 839, 857, 907, 981, 1105, 1141) becomes `fit(R, p, n)`
- Test: the existing suite (no new test; this is a signature change verified by the suite staying green)

**Interfaces:**
- Consumes: `fit(rng::AbstractRNG, problem, n_iterations)` (already exists, vi.jl:524).
- Produces: `fit(problem::VariationalProblem, n_iterations::Integer) -> AbstractVariationalDistribution` (auto rng). The `fit(problem, n; rng=…)` keyword method is **removed**. The positional `fit(rng, problem, n)` is unchanged.

- [ ] **Step 1: Replace the `fit` keyword method**

In `src/vi.jl`, replace line 531-532 (`fit(problem::VariationalProblem, n_iterations::Integer; rng::AbstractRNG = …) = fit(rng, problem, n_iterations)`) with:

```julia
fit(problem::VariationalProblem, n_iterations::Integer) =
    fit(Random.default_rng(), problem, n_iterations)
```

If the `fit(rng, problem, n)` method just above it carries a docstring mentioning `; rng`, update that prose to `fit([rng], problem, n)`; otherwise leave the method body untouched.

- [ ] **Step 2: Run tests to verify they now FAIL on the old call sites**

Run: `julia --project -e 'include("test/runtests.jl")'`
Expected: FAIL — `fit(::VariationalProblem, ::Int; rng=…)` no longer has a method (keyword removed); errors at the first `fit(...; rng=...)` call site.

- [ ] **Step 3: Convert every `fit(p, n; rng = R)` call site to `fit(R, p, n)`**

Apply this mechanical rewrite across `test/runtests.jl`. Each occurrence of the form `fit(<probexpr>, <n>; rng = <rngexpr>)` becomes `fit(<rngexpr>, <probexpr>, <n>)`. The 13 sites:

```
310:  @test_throws ArgumentError fit(nd_problem, 1; rng = MersenneTwister(1))
        -> @test_throws ArgumentError fit(MersenneTwister(1), nd_problem, 1)
596:  mgvi_post   = fit(mgvi_problem, 3; rng = MersenneTwister(5))
        -> fit(MersenneTwister(5), mgvi_problem, 3)
628:  geovi_post  = fit(geovi_problem, 3; rng = MersenneTwister(5))
        -> fit(MersenneTwister(5), geovi_problem, 3)
642:  adam_post   = fit(adam_problem, 400; rng = MersenneTwister(11))
        -> fit(MersenneTwister(11), adam_problem, 400)
713:  post        = fit(geovi, 8; rng = MersenneTwister(9))
        -> fit(MersenneTwister(9), geovi, 8)
778:  post        = fit(problem, 6; rng = MersenneTwister(2))
        -> fit(MersenneTwister(2), problem, 6)
808:  post        = fit(problem, 8; rng = MersenneTwister(2025))
        -> fit(MersenneTwister(2025), problem, 8)
839:  coupled_post= fit(coupled_problem, 8; rng = MersenneTwister(2025))
        -> fit(MersenneTwister(2025), coupled_problem, 8)
857:  delta_post  = fit(delta_problem, 8; rng = MersenneTwister(2025))
        -> fit(MersenneTwister(2025), delta_problem, 8)
907:  post        = fit(mf, 3000; rng = MersenneTwister(0xfeed))
        -> fit(MersenneTwister(0xfeed), mf, 3000)
981:  post        = fit(problem, 3000; rng = MersenneTwister(0x01))
        -> fit(MersenneTwister(0x01), problem, 3000)
1105: post        = fit(problem, 8; rng = MersenneTwister(0xfeed))
        -> fit(MersenneTwister(0xfeed), problem, 8)
1141: mf_post     = fit(mf_problem, 3000; rng = MersenneTwister(0xabcd))
        -> fit(MersenneTwister(0xabcd), mf_problem, 3000)
```

After editing, confirm none remain:

Run: `grep -rn "fit(.*; *rng" test/`
Expected: no output.

- [ ] **Step 4: Run the full suite to verify it passes**

Run: `julia --project -e 'using Pkg; Pkg.test()'`
Expected: PASS — all testsets green (the conversions are behavior-preserving; `fit(R, p, n)` and `fit(p, n; rng=R)` previously produced identical results).

- [ ] **Step 5: Commit**

```bash
git add src/vi.jl test/runtests.jl
git commit -m "refactor: fit rng is positional-or-auto, not a keyword"
```

---

### Task 4: Documentation prose (docs/ + CLAUDE.md)

**Files:**
- Modify: `docs/src/design.md:253` and `:282`
- Modify: `docs/src/interfaces.md:230`
- Modify: `CLAUDE.md` (lines 45, 109, 125 — `fit(problem, n; rng)` mentions and examples)
- Test: none (documentation only; verified by inspection / no broken doctests)

**Interfaces:**
- Consumes: the final `init`/`reset!`/`fit` signatures from Tasks 1-3.
- Produces: nothing code-facing.

- [ ] **Step 1: Update `docs/src/design.md`**

Line 253: change `` `fit(problem, n; rng)` is a convenience that runs the loop and returns the posterior.`` to `` `fit([rng], problem, n)` is a convenience that runs the loop and returns the posterior (rng positional or auto).``

Line 282: change the example `q     = fit(problem, 8; rng)        # returns the fitted variational distribution` to `q     = fit(rng, problem, 8)        # returns the fitted variational distribution`.

If `design.md` documents `init` nearby, add a one-line mention that `init([rng], problem[, ξ0])` accepts an optional starting point and that `reset!(state, problem, ξ0)` restarts an existing state in place.

- [ ] **Step 2: Update `docs/src/interfaces.md`**

Line 230: change `` `fit(problem, n; rng)` is a convenience that runs the loop and returns the fitted`` to `` `fit([rng], problem, n)` is a convenience that runs the loop and returns the fitted`` (rng positional or auto). In the same interfaces doc, add `reset!` next to `init` in the loop-interface description: "`reset!(state, problem, ξ0)` re-initializes an existing state in place from a new starting point (reuses buffers; Reactant-safe)."

- [ ] **Step 3: Update `CLAUDE.md`**

- Line 45: change `` `fit(problem, n; rng)` is a convenience`` to `` `fit([rng], problem, n)` is a convenience`` (rng positional or auto).
- Line 109: `q = fit(problem, 8; rng=MersenneTwister(42))` → `q = fit(MersenneTwister(42), problem, 8)`.
- Line 125: `q = fit(problem, 3000; rng=MersenneTwister(0))` → `q = fit(MersenneTwister(0), problem, 3000)`.
- In the "Typical usage pattern" block (around line 95-135), add after the `init(...)`/`fit(...)` lines a one-liner: `# restart from a new point without rebuilding `problem`:` then `rng, state = init(MersenneTwister(42), problem, xi0_new)   # or: reset!(state, problem, xi0_new)`.

- [ ] **Step 4: Verify no stale keyword references remain**

Run: `grep -rn "fit(.*; *rng\|init(.*; *rng" docs/ CLAUDE.md README.md src/`
Expected: no output (all converted; the spec file under `docs/superpowers/specs/` is allowed to retain historical references — exclude it: `grep -rn "fit(.*; *rng\|init(.*; *rng" docs/src CLAUDE.md README.md src/`).

- [ ] **Step 5: Commit**

```bash
git add docs/src/design.md docs/src/interfaces.md CLAUDE.md
git commit -m "docs: rng positional-or-auto for init/fit; document ξ0 and reset!"
```

---

## Self-Review

**Spec coverage:**
- Section A (init optional ξ0, rng positional-or-auto) → Task 1. ✓
- Section B (`reset!` in place, optimizer reassigned, size guard) → Task 2. ✓
- `fit` rng keyword removal + call-site churn → Task 3. ✓
- Exports (`reset!`) → Task 2 Step 4. ✓
- Docstring updates (init/fit/reset!) → Task 1 Step 3, Task 2 Step 3, Task 3 Step 1. ✓
- Prose docs (design.md/interfaces.md/CLAUDE.md) → Task 4. ✓
- Testing bullets (init from ξ0 for both families; init unchanged; reset! in-place identity + zeroed optimizer; size error; mean-field NamedTuple θ) → Task 1/2 tests. ✓
- Reactant identity guarantee → covered by the `===` identity assertions in Task 2 (compile-and-rerun left as CI-conditional per spec; not scripted here since Reactant is an optional dep). ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; every command shows expected output. ✓

**Type consistency:** `reset!(state, problem, ξ0::AbstractArray) -> state` used identically in Task 2 interface, implementation, and tests. `init(rng, problem, ξ0=nothing)` consistent across Task 1 interface/impl/tests. `_latent_ref` defined once (Task 2 Step 3) and used only there. `fit(rng, problem, n)` (positional) is the pre-existing method all Task 3 conversions target. ✓

**One known spec-vs-plan deviation (intentional, already reflected in the committed spec):** optimizer state is *reassigned*, not `fmap(copyto!)`-ed — the spec was corrected to match before this plan was written.

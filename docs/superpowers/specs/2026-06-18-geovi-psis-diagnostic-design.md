# GeoVI PSIS Diagnostic — Design

**Date:** 2026-06-18
**Status:** Approved (design); ready for implementation plan
**Repo:** GeoVI (`vi-api-redesign` branch)

## Motivation

Fitting a Comrade + GeoVI VLBI imaging posterior gives partially inconsistent
results across runs. We want a *diagnostic* for the quality of the variational
approximation — how well the fitted `q` covers the true posterior `p` — in the
spirit of Pathfinder's Pareto-smoothed importance sampling (PSIS). The Pareto
shape parameter `k̂` is the headline number:

- `k̂ < 0.5` — good; the variational approximation covers the posterior well.
- `0.5 ≤ k̂ < 0.7` — usable but imperfect.
- `k̂ ≥ 0.7` — the approximation is missing posterior mass; importance-corrected
  estimates are unreliable.

As a bonus, the Pareto-smoothed self-normalized weights let us compute
importance-corrected posterior expectations (means, χ², images).

## The binding constraint and how we get around it

PSIS needs importance weights `w_i = p(ξ_i) / q(ξ_i)`, i.e. the proposal density
`log q(ξ)`. GeoVI/MGVI deliberately expose **no** tractable `logdensity`
(`interface.jl:188`, `fisher_gaussian.jl:107`): the normalization is the
log-determinant `½ log det(I + F)`, which the code holds constant.

**Key fact that unblocks PSIS:** for a single fitted `q` the metric is pinned at
one mean `μ`, so `½ log det(I + F(μ))` is the *same additive constant for every
draw*. The Pareto shape `k̂` is invariant to scaling all weights by a constant
(the GPD is closed under scaling), and self-normalized reweighting cancels the
same constant. **So we never need the log-determinant.** We compute `log q(ξ)`
*up to that shared constant* and both `k̂` and the reweighting come out exact.

### Exactness by family

- **MGVI:** `log q(ξ) = -½ (ξ-μ)ᵀ (I+F(μ)) (ξ-μ) + const`. The proposal is exactly
  `N(μ, (I+F(μ))⁻¹)` (the draw linearizes `F` at `μ`), so this is the exact
  Gaussian log-density up to the cancelling constant. **`k̂` is exact.**
- **geoVI:** the draw bends each sample through a per-sample nonlinear curve, so
  its true density is a pushforward whose log-Jacobian differs *per sample* and
  does **not** cancel. We use the same metric-Gaussian density, which is an
  **approximate** diagnostic: it measures how well the underlying metric-Gaussian
  (the MGVI proposal) covers the posterior, ignoring the curve correction. This
  is documented in the API, not hidden.

(We deliberately do **not** implement the exact normalized density. It is
computable for MGVI via the matrix determinant lemma —
`log det(I_D + RᵀR) = log det(I_M + RRᵀ)`, an `M×M` determinant where
`M = data dimension ≪ D = latent dimension`, costing `M` linearization applies +
`O(M³)` — but it is unnecessary for `k̂` and out of scope. See "Future work".)

## Design

Three independently testable layers.

### Layer 1 — `logdensity_unnormalized(d, ξ)` (the primitive)

New capability on `AbstractVariationalDistribution`:

```julia
# Generic fallback: a tractable normalized density already works for PSIS.
logdensity_unnormalized(d::AbstractVariationalDistribution, ξ) = logdensity(d, ξ)

# Fisher-Gaussian: the metric-Gaussian log-density up to the (sample-independent)
# ½logdet(I+F(μ)) constant. One `_posterior_metric` matvec.
function logdensity_unnormalized(d::FisherGaussianDistribution, ξ)
    δ = ξ .- d.mean
    return -0.5 * real(dot(δ, _posterior_metric(d.likelihood, d.mean, δ)))
end
```

Exact for MGVI; the underlying metric-Gaussian for geoVI (curve ignored) —
documented in the docstring. Runs on the device likelihood ⇒ Reactant-safe.

### Layer 2 — `log_importance_ratio(d, ξ)` (per-sample compilable scalar)

```julia
log_importance_ratio(d::AbstractVariationalDistribution, ξ) =
    -_negative_logposterior(d.likelihood, ξ) - logdensity_unnormalized(d, ξ)
```

`= log p(ξ) − log q(ξ)` up to the cancelling constant. Pure scalar over the
device likelihood; the user `@compile`s this and maps it over draws.

### Layer 3 — `pareto_diagnostic` (host, wraps PSIS.jl)

```julia
# Core: takes precomputed log-ratios (the Reactant path).
pareto_diagnostic(log_ratios::AbstractVector) -> PSIS.PSISResult

# Convenience: draws + computes internally (non-Reactant / host use).
pareto_diagnostic(rng, d::AbstractVariationalDistribution, n::Integer) -> PSIS.PSISResult
```

Thin wrapper over `PSIS.psis`. Returns the PSIS result carrying `k̂`
(`pareto_shape`) and the Pareto-smoothed self-normalized `weights`. (Exact field
names confirmed against the installed PSIS.jl version at implementation.)

### Driver usage (reuses existing draws — no second sampling pass)

```julia
lr = @compile sync=true GeoVI.log_importance_ratio(q, ξ0)
logw = [Float64(lr(q, Reactant.to_rarray(ξ))) for ξ in ξs]
psis = GeoVI.pareto_diagnostic(logw)
@info "PSIS" k̂=psis.pareto_shape          # <0.5 good · 0.5–0.7 ok · >0.7 q misses mass
# importance-corrected estimates: sum(psis.weights .* f.(ps))
```

## Error handling

- Degenerate `q ≈ p` (near-constant log-ratios) ⇒ GPD fit degenerate. Surface
  `k̂` as whatever PSIS.jl returns (`-Inf`/`NaN`) cleanly as "excellent fit";
  do not crash.
- Reweighting docs state plainly: `k̂ > 0.7` ⇒ importance-corrected estimates are
  unreliable (the diagnostic's purpose).
- geoVI docstring states the diagnostic is the metric-Gaussian approximation
  (curve ignored), not the exact geoVI density.

## Testing (GeoVI suite)

1. `logdensity_unnormalized` on a linear-Gaussian likelihood with known `F`:
   equals the analytic Gaussian log-density up to a constant (check pairwise
   differences across points).
2. PSIS end-to-end: perfect proposal (`q == p`) ⇒ `k̂ ≤ 0`, near-uniform weights;
   deliberately narrowed `q` ⇒ large `k̂`.
3. Constant-shift invariance: shifting all `log_ratios` by a constant leaves `k̂`
   and the normalized weights unchanged.
4. If the suite runs Reactant: compile `log_importance_ratio` and check
   host == device.

## Dependencies

Add `PSIS` (Seth Axen — same author as Pathfinder.jl) to GeoVI `Project.toml`
and `using PSIS` in the package.

## Decisions (locked)

| Decision | Choice |
|---|---|
| geoVI semantics | Gaussian-metric `k̂`, documented as approximate (curve ignored) |
| Output | `k̂` + Pareto-smoothed weights (full reweighting) |
| PSIS implementation | Depend on PSIS.jl |
| Location | GeoVI package feature |
| Normalized `log q` | Unnormalized only (constant cancels in `k̂`) |

## Future work (out of scope)

Exact normalized `logdensity(d::FisherGaussianDistribution, ξ)` for MGVI via the
low-rank determinant lemma (`log det(I_M + RRᵀ)`, `M` linearization applies +
`O(M³)` Cholesky), enabling a proper ELBO and cross-iteration / cross-mean
density comparison. Not needed for `k̂`. geoVI would still fall back to
unnormalized (the curve pushforward Jacobian is not the metric determinant).

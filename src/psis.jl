# ── PSIS diagnostic ──────────────────────────────────────────────────────────
# Pareto-smoothed importance sampling on a fitted variational distribution. The Pareto
# shape `k̂` diagnoses how well `q` covers the posterior `p`: k̂ < 0.5 good, 0.5–0.7 ok,
# ≥ 0.7 the approximation misses posterior mass and importance-corrected estimates are
# unreliable. Builds on `log_importance_ratio` (unnormalized log p − log q); the dropped
# ½logdet(I+F) constant cancels in both the GPD shape estimate and the self-normalized
# weights, so the diagnostic and the reweighting are exact (for MGVI) without it.

"""
    pareto_diagnostic(log_ratios::AbstractVector) -> NamedTuple

Run PSIS on precomputed unnormalized log importance ratios (e.g. a compiled
[`log_importance_ratio`](@ref) mapped over device draws). Returns
`(; pareto_shape, weights, result)`: the Pareto `k̂`, the normalized Pareto-smoothed
importance weights, and the full `PSIS.PSISResult`.
"""
function pareto_diagnostic(log_ratios::AbstractVector)
    result = PSIS.psis(collect(float.(log_ratios)); warn = false)
    return (; pareto_shape = result.pareto_shape, weights = result.weights, result)
end

"""
    pareto_diagnostic(rng, d::FisherGaussianDistribution, n) -> NamedTuple

Draw `n` samples from `d`, compute their log importance ratios, and run PSIS. Convenience
for host (non-Reactant) use; under Reactant, compile `log_importance_ratio` and pass the
resulting vector to the `pareto_diagnostic(log_ratios)` method instead.
"""
function pareto_diagnostic(rng::AbstractRNG, d::FisherGaussianDistribution, n::Integer)
    n > 0 || throw(ArgumentError("`n` must be positive"))
    draws = rand(rng, d, n)
    log_ratios = [log_importance_ratio(d, _sample_slice(draws, i)) for i in 1:n]
    return pareto_diagnostic(log_ratios)
end

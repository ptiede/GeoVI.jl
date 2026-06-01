randn_like(rng::AbstractRNG, x::AbstractArray) =
    randn(rng, eltype(x), size(x))

# ── Tree-generic parameter ops ──────────────────────────────────────────────
#
# The variational parameter container θ held by `VIState.position` is either a
# bare array (MGVI/geoVI, where θ *is* the latent mean) or a Functors-compatible
# structure such as the NamedTuple `(; mean, logstd)` of mean-field ADVI. These
# helpers act uniformly on both; for a bare array each reduces to the obvious
# array operation, so the MGVI/geoVI numerics are unchanged.

# Elementwise difference `a - b` over the tree (whole-array `-` per leaf).
_param_sub(a::AbstractArray, b::AbstractArray) = a .- b
_param_sub(a, b) = fmap(_param_sub, a, b)

# Euclidean norm over all leaves (√ of the summed per-leaf squared norms).
_param_norm(x::AbstractArray) = norm(x)
_param_norm(x) = sqrt(sum(v -> abs2(_param_norm(v)), Optimisers.trainables(x)))

# `all(isfinite, ·)` over every leaf. `mapreduce(…, &)` (not short-circuiting
# `all`) so the per-leaf reduction also works on traced `Bool`s under Reactant.
_param_all_finite(x::AbstractArray) = all(isfinite, x)
_param_all_finite(x) = mapreduce(_param_all_finite, &, Optimisers.trainables(x))
